import math
import time
from tqdm import tqdm
from retrieval.querySearch import QuerySearch
from test.evalManifestReader import EvalManifestReader, VideoEvalEntry
from test.evalMetrics import MetricsCalculator, EvalMetrics


class Evaluator:
    def __init__(self, config, visionEncoder):
        self.querySearch = QuerySearch(config, visionEncoder)
        self.metricsCalculator = MetricsCalculator()

    def _confidenceFromScores(self, scores: list) -> float:
        sampleCount = len(scores)
        if sampleCount <= 1:
            return 1.0
        positiveScores = [max(s, 0.0) for s in scores]
        total = sum(positiveScores)
        if total <= 1e-12:
            return 0.0
        probabilities = [s / total for s in positiveScores if s > 0]
        rawEntropy = -sum(p * math.log(p) for p in probabilities)
        normalizedEntropy = rawEntropy / math.log(sampleCount)
        return 1.0 - normalizedEntropy

    def _findRank(self, results: list, targetVideoId: str) -> int | None:
        for rank, result in enumerate(results, start=1):
            if result["videoId"] == targetVideoId:
                return rank
        return None

    def _evaluateSingleEntry(self, entry: VideoEvalEntry) -> dict:
        query = entry.captions[0]
        startTime = time.perf_counter()
        results, _ = self.querySearch.queryWithText(query, topK=10, retrievalMode="both")
        queryTimeSec = time.perf_counter() - startTime

        scores = [r["score"] for r in results]
        confidence = self._confidenceFromScores(scores)
        rank = self._findRank(results, entry.videoId)

        return {
            "videoId": entry.videoId,
            "query": query,
            "rank": rank,
            "queryTimeSec": queryTimeSec,
            "confidence": confidence,
        }

    def run(self, csvPath: str) -> EvalMetrics:
        entries = EvalManifestReader(csvPath).read()
        queryResults = []
        progressBar = tqdm(entries, desc="Evaluating", unit="video")
        for entry in progressBar:
            progressBar.set_postfix_str(entry.videoId)
            queryResults.append(self._evaluateSingleEntry(entry))
        return self.metricsCalculator.compute(queryResults)