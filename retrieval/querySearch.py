import time
import numpy as np
from storage.indexStore import IndexStore
from storage.pcaReducer import PcaDimensionReducer, IdentityDimensionReducer
from retrieval.queryStatsReporter import QueryStatsReporter

allowedModalitiesByRetrievalMode = {
    "audio": {"transcript"},
    "image": {"frame"},
    "both": {"transcript", "frame"},
}


class QuerySearch:
    def __init__(self, config, visionEncoder):
        self.config = config
        self.visionEncoder = visionEncoder
        self.indexStore = IndexStore(config.indexDir)
        self.queryStatsReporter = QueryStatsReporter(config.statsDir)

    def queryWithText(self, textQuery: str, topK: int = 5, retrievalMode: str = None, pcaDim: int = None):
        return self._query(lambda: self.visionEncoder.encodeText(textQuery), topK, retrievalMode, f"text:{textQuery}", pcaDim)

    def queryWithImage(self, frame, topK: int = 5, retrievalMode: str = None, queryLabel: str = "image", pcaDim: int = None):
        return self._query(lambda: self.visionEncoder.encodeImage(frame), topK, retrievalMode, queryLabel, pcaDim)

    def _query(self, embedFn, topK: int, retrievalMode: str, queryLabel: str, pcaDim: int = None):
        timings = {}
        retrievalMode = retrievalMode or self.config.retrievalMode
        allowedModalities = allowedModalitiesByRetrievalMode[retrievalMode]

        startTime = time.perf_counter()
        rawQuery = embedFn()
        timings["embedQuerySec"] = time.perf_counter() - startTime

        activeDim = pcaDim if pcaDim is not None else (self.config.pcaDimension if self.config.usePcaReduction else None)

        startTime = time.perf_counter()
        reducedFeatures = self.indexStore.loadReducedFeatures(dimension=activeDim)
        metadata = self.indexStore.loadMetadata()
        timings["loadIndexSec"] = time.perf_counter() - startTime

        startTime = time.perf_counter()
        if activeDim is not None:
            reducer = PcaDimensionReducer.load(self.indexStore.indexDir, activeDim)
        else:
            reducer = IdentityDimensionReducer()
        transformedQuery = reducer.transform(rawQuery.reshape(1, -1)).flatten()
        timings["transformQuerySec"] = time.perf_counter() - startTime

        startTime = time.perf_counter()
        normalizedQuery = transformedQuery / (np.linalg.norm(transformedQuery) + 1e-10)
        normalizedIndex = reducedFeatures / (np.linalg.norm(reducedFeatures, axis=1, keepdims=True) + 1e-10)
        similarities = normalizedIndex @ normalizedQuery
        weights = np.array([item.get("appliedWeight", 1.0) for item in metadata])
        modalityMask = np.array([item["modality"] in allowedModalities for item in metadata])
        weightedSimilarities = np.where(modalityMask, similarities * weights, -np.inf)
        timings["similaritySec"] = time.perf_counter() - startTime

        startTime = time.perf_counter()
        rankedIndices = np.argsort(weightedSimilarities)[::-1]
        results, seenVideoIds = [], set()
        for index in rankedIndices:
            if weightedSimilarities[index] == -np.inf:
                break
            videoId = metadata[index]["videoId"]
            if videoId in seenVideoIds:
                continue
            seenVideoIds.add(videoId)
            results.append({
                "videoId": videoId,
                "score": float(weightedSimilarities[index]),
                "matchedItem": metadata[index],
            })
            if len(results) >= topK:
                break
        timings["rankAndDedupeSec"] = time.perf_counter() - startTime

        timings["totalSec"] = sum(timings.values())
        self.queryStatsReporter.recordQuery(queryLabel, retrievalMode, results, timings)
        return results, timings