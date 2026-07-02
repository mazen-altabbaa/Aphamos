import math
import time
import numpy as np
import torch
from tqdm import tqdm
from storage.indexStore import IndexStore
from storage.pcaReducer import PcaDimensionReducer, IdentityDimensionReducer
from test.evalManifestReader import EvalManifestReader, VideoEvalEntry
from test.evalMetrics import MetricsCalculator, EvalMetrics

PCA_DIMENSIONS = [512, 256, 128, 64]
RETRIEVAL_MODES = ["both", "audio", "image"]
MODALITY_SETS = {
    "both": {"transcript", "frame"},
    "audio": {"transcript"},
    "image": {"frame"},
}
TOP_K = 10
ENCODE_BATCH_SIZE = 64


class IndexCache:
    def __init__(self, config):
        self.indexStore = IndexStore(config.indexDir)
        self.metadata = self.indexStore.loadMetadata()
        self.videoIds = np.array([m["videoId"] for m in self.metadata])
        self.modalities = np.array([m["modality"] for m in self.metadata])
        self.weights = np.array([m.get("appliedWeight", 1.0) for m in self.metadata])

        self.reducedFeatures = {}
        self.reducers = {}
        print("Preloading index for all PCA dimensions...")
        for dim in PCA_DIMENSIONS:
            features = self.indexStore.loadReducedFeatures(dimension=dim)
            if features is not None:
                norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-10
                self.reducedFeatures[dim] = (features / norm).astype(np.float32)
                self.reducers[dim] = PcaDimensionReducer.load(self.indexStore.indexDir, dim)
        print(f"  Loaded {len(self.reducedFeatures)} PCA variants, {len(self.metadata)} segments.")


class Evaluator:
    def __init__(self, config, visionEncoder):
        self.config = config
        self.visionEncoder = visionEncoder
        self.metricsCalculator = MetricsCalculator()

    def _encodeAllCaptions(self, entries: list) -> tuple:
        allCaptions = []
        entryCaption_indices = []
        for entry in entries:
            indices = []
            for caption in entry.captions:
                indices.append(len(allCaptions))
                allCaptions.append(caption)
            entryCaption_indices.append(indices)

        print(f"Encoding {len(allCaptions)} captions on GPU in batches of {ENCODE_BATCH_SIZE}...")
        allEmbeddings = []
        for i in tqdm(range(0, len(allCaptions), ENCODE_BATCH_SIZE), desc="Encoding captions", unit="batch"):
            batch = allCaptions[i:i + ENCODE_BATCH_SIZE]
            batchEmbeddings = np.stack([self.visionEncoder.encodeText(c) for c in batch])
            allEmbeddings.append(batchEmbeddings)

        rawEmbeddings = np.vstack(allEmbeddings).astype(np.float32)
        return rawEmbeddings, entryCaption_indices

    def _transformEmbeddings(self, rawEmbeddings: np.ndarray, dim: int) -> np.ndarray:
        reducer = self.indexCache.reducers.get(dim)
        if reducer is None:
            return None
        transformed = reducer.transform(rawEmbeddings)
        norm = np.linalg.norm(transformed, axis=1, keepdims=True) + 1e-10
        return (transformed / norm).astype(np.float32)

    def _confidenceFromScores(self, scores: np.ndarray) -> float:
        positiveScores = np.maximum(scores, 0.0)
        total = positiveScores.sum()
        if total <= 1e-12:
            return 0.0
        probs = positiveScores / total
        probs = probs[probs > 0]
        rawEntropy = -np.sum(probs * np.log(probs))
        normalizedEntropy = rawEntropy / math.log(len(scores))
        return float(1.0 - normalizedEntropy)

    def _computeAllResults(self, queryEmbeddings: np.ndarray, mode: str,
                           features: np.ndarray) -> np.ndarray:
        modalityMask = np.array([m in MODALITY_SETS[mode] for m in self.indexCache.modalities])
        maskedWeights = np.where(modalityMask, self.indexCache.weights, 0.0).astype(np.float32)
        similarities = queryEmbeddings @ features.T
        return similarities * maskedWeights[np.newaxis, :]

    def _findBestRank(self, weightedSims: np.ndarray, targetVideoId: str,
                      captionIndices: list) -> tuple:
        bestRank = None
        bestConfidence = 0.0

        for capIdx in captionIndices:
            sims = weightedSims[capIdx]
            sortedIndices = np.argsort(sims)[::-1]
            seen = {}
            rank = 0
            for idx in sortedIndices:
                if sims[idx] <= 0:
                    break
                vid = self.indexCache.videoIds[idx]
                if vid not in seen:
                    seen[vid] = True
                    rank += 1
                    if rank >= TOP_K:
                        break
            topVideoIds = list(seen.keys())
            topScores = np.array([sims[np.where(self.indexCache.videoIds == v)[0]].max()
                                  for v in topVideoIds])
            if targetVideoId in topVideoIds:
                r = topVideoIds.index(targetVideoId) + 1
                conf = self._confidenceFromScores(topScores[:TOP_K])
                if bestRank is None or r < bestRank:
                    bestRank = r
                    bestConfidence = conf

        return bestRank, bestConfidence

    def _findBestRankFast(self, weightedSims: np.ndarray, entries: list,
                          entryCaption_indices: list) -> list:
        videoIds = self.indexCache.videoIds

        results = []
        for entryIdx, entry in enumerate(entries):
            captionIndices = entryCaption_indices[entryIdx]
            bestRank = None
            bestConfidence = 0.0

            for capIdx in captionIndices:
                sims = weightedSims[capIdx]
                sortedIdx = np.argsort(sims)[::-1]
                seen = {}
                rank = 0
                for i in sortedIdx:
                    if sims[i] <= 0:
                        break
                    vid = videoIds[i]
                    if vid not in seen:
                        seen[vid] = rank
                        rank += 1
                    if rank >= TOP_K:
                        break

                if entry.videoId in seen:
                    r = seen[entry.videoId] + 1
                    topScores = sims[[sortedIdx[j] for j in range(min(TOP_K * 5, len(sortedIdx)))]]
                    conf = self._confidenceFromScores(topScores[:TOP_K])
                    if bestRank is None or r < bestRank:
                        bestRank = r
                        bestConfidence = conf

            results.append({
                "videoId": entry.videoId,
                "rank": bestRank,
                "queryTimeSec": 0.0,
                "confidence": bestConfidence,
            })
        return results

    def runAll(self, csvPath: str) -> dict:
        entries = EvalManifestReader(csvPath).read()
        self.indexCache = IndexCache(self.config)

        rawEmbeddings, entryCaption_indices = self._encodeAllCaptions(entries)

        allResults = {}
        totalCombinations = len(PCA_DIMENSIONS) * len(RETRIEVAL_MODES)
        comboBar = tqdm(total=totalCombinations, desc="Evaluating combinations", unit="combo")

        for dim in PCA_DIMENSIONS:
            if dim not in self.indexCache.reducedFeatures:
                for mode in RETRIEVAL_MODES:
                    comboBar.update(1)
                continue

            queryEmbeddings = self._transformEmbeddings(rawEmbeddings, dim)
            features = self.indexCache.reducedFeatures[dim]

            for mode in RETRIEVAL_MODES:
                label = f"pca{dim}_{mode}"
                comboBar.set_description(f"Evaluating {label}")

                startTime = time.perf_counter()
                weightedSims = self._computeAllResults(queryEmbeddings, mode, features)
                queryResults = self._findBestRankFast(weightedSims, entries, entryCaption_indices)
                elapsed = time.perf_counter() - startTime

                for r in queryResults:
                    r["queryTimeSec"] = elapsed / len(entries)

                allResults[label] = self.metricsCalculator.compute(queryResults)
                comboBar.update(1)

        comboBar.close()
        return allResults