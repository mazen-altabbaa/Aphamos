import ast
import csv
from pathlib import Path
from dataclasses import dataclass


@dataclass
class VideoEvalEntry:
    videoId: str
    captions: list


class EvalManifestReader:
    def __init__(self, csvPath: str):
        self.csvPath = Path(csvPath)

    def _parseCaptions(self, raw: str) -> list:
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed if str(c).strip()]
        except Exception:
            pass
        return [raw.strip()] if raw.strip() else []

    def read(self) -> list:
        entries = []
        with open(self.csvPath, newline="", encoding="utf-8") as csvFile:
            reader = csv.DictReader(csvFile)
            for row in reader:
                videoId = row.get("videoID", "").strip()
                captions = self._parseCaptions(row.get("caption", ""))
                if videoId and captions:
                    entries.append(VideoEvalEntry(videoId=videoId, captions=captions))
        return entries