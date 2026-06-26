from abc import ABC, abstractmethod
from pathlib import Path
import csv


class IDatasetLoader(ABC):
    @abstractmethod
    def listVideoPaths(self) -> list:
        ...


class LocalFolderDatasetLoader(IDatasetLoader):
    videoExtensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm")

    def __init__(self, folderPath: str):
        self.folderPath = Path(folderPath)

    def listVideoPaths(self) -> list:
        return sorted(
            str(path) for extension in self.videoExtensions
            for path in self.folderPath.glob(f"*{extension}")
        )


class CsvManifestDatasetLoader(IDatasetLoader):
    def __init__(self, csvPath: str, pathColumn: str = "videoPath"):
        self.csvPath = Path(csvPath)
        self.pathColumn = pathColumn

    def listVideoPaths(self) -> list:
        with open(self.csvPath, newline="", encoding="utf-8") as csvFile:
            reader = csv.DictReader(csvFile)
            return [row[self.pathColumn] for row in reader if Path(row[self.pathColumn]).exists()]
