import json
import time
from pathlib import Path
import matplotlib.pyplot as plt


class StatsReporter:
    def __init__(self, statsDir: Path):
        self.statsDir = Path(statsDir)
        self.statsDir.mkdir(parents=True, exist_ok=True)
        self.summaryPath = self.statsDir / "summary.json"

    def loadSummary(self) -> dict:
        if self.summaryPath.exists():
            return json.loads(self.summaryPath.read_text())
        return {"totalProcessingTimeSec": 0.0, "totalVideosIndexed": 0, "collections": []}

    def saveSummary(self, summary: dict) -> None:
        self.summaryPath.write_text(json.dumps(summary, indent=2))

    def reportCollection(self, collectionName: str, videoStatsList: list, dimensionReducer, isPcaUsed: bool):
        summary = self.loadSummary()
        collectionTimeSec = sum(item["processingTimeSec"] for item in videoStatsList)
        confidenceCoefficient = dimensionReducer.explainedVarianceRatioSum() if isPcaUsed else None

        collectionRecord = {
            "collectionName": collectionName,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "videoCount": len(videoStatsList),
            "totalTimeSec": collectionTimeSec,
            "averageTimePerVideoSec": collectionTimeSec / max(len(videoStatsList), 1),
            "confidenceCoefficient": confidenceCoefficient,
            "perVideo": [
                {
                    "videoId": item["videoId"],
                    "durationSec": item["durationSec"],
                    "capturedFrameCount": item["capturedFrameCount"],
                    "processingTimeSec": item["processingTimeSec"],
                    "audioWeight": item["audioWeight"],
                    "frameWeight": item["frameWeight"],
                }
                for item in videoStatsList
            ],
        }

        summary["collections"].append(collectionRecord)
        summary["totalProcessingTimeSec"] += collectionTimeSec
        summary["totalVideosIndexed"] += len(videoStatsList)
        self.saveSummary(summary)

        self._plotThresholdHistory(collectionName, videoStatsList)
        self._plotCumulativeTime(summary)
        return collectionRecord

    def _plotThresholdHistory(self, collectionName: str, videoStatsList: list) -> None:
        figure, axis = plt.subplots(figsize=(8, 5))
        for item in videoStatsList:
            if item["thresholdHistory"]:
                axis.plot(item["thresholdHistory"], label=item["videoId"])
        axis.set_xlabel("Captured frame index")
        axis.set_ylabel("Adaptive threshold value")
        axis.set_title(f"Threshold evolution - {collectionName}")
        if len(videoStatsList) <= 12:
            axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(self.statsDir / f"{collectionName}_thresholdHistory.png")
        plt.close(figure)

    def _plotCumulativeTime(self, summary: dict) -> None:
        figure, axis = plt.subplots(figsize=(8, 5))
        names = [collection["collectionName"] for collection in summary["collections"]]
        times = [collection["totalTimeSec"] for collection in summary["collections"]]
        axis.bar(names, times)
        axis.set_ylabel("Total indexing time (sec)")
        axis.set_title("Indexing time per collection")
        figure.tight_layout()
        figure.savefig(self.statsDir / "cumulativeIndexingTime.png")
        plt.close(figure)
