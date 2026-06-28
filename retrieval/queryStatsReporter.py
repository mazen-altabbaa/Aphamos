import json
import math
import time
from pathlib import Path


class QueryStatsReporter:
    def __init__(self, statsDir: Path):
        self.statsDir = Path(statsDir)
        self.statsDir.mkdir(parents=True, exist_ok=True)
        self.summaryPath = self.statsDir / "queryStatsSummary.json"

    def loadSummary(self) -> dict:
        if self.summaryPath.exists():
            return json.loads(self.summaryPath.read_text())
        return {"totalQueries": 0, "queries": []}

    def saveSummary(self, summary: dict) -> None:
        self.summaryPath.write_text(json.dumps(summary, indent=2))

    def recordQuery(self, queryLabel: str, retrievalMode: str, results: list, timings: dict) -> dict:
        summary = self.loadSummary()
        scores = [result["score"] for result in results]
        confidenceCoefficient = self._confidenceFromScores(scores)

        queryRecord = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "queryLabel": queryLabel,
            "retrievalMode": retrievalMode,
            "resultCount": len(results),
            "topVideoId": results[0]["videoId"] if results else None,
            "topScore": scores[0] if scores else None,
            "confidenceCoefficient": confidenceCoefficient,
            "timings": timings,
        }

        summary["queries"].append(queryRecord)
        summary["totalQueries"] += 1
        self.saveSummary(summary)
        return queryRecord

    def _confidenceFromScores(self, scores: list) -> float:
        sampleCount = len(scores)
        if sampleCount <= 1:
            return 1.0

        positiveScores = [max(score, 0.0) for score in scores]
        total = sum(positiveScores)
        if total <= 1e-12:
            return 0.0

        probabilities = [score / total for score in positiveScores if score > 0]
        rawEntropy = -sum(probability * math.log(probability) for probability in probabilities)
        normalizedEntropy = rawEntropy / math.log(sampleCount)
        return 1.0 - normalizedEntropy
