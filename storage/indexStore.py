import json
from pathlib import Path
import numpy as np


class IndexStore:
    def __init__(self, indexDir: Path):
        self.indexDir = Path(indexDir)
        self.indexDir.mkdir(parents=True, exist_ok=True)
        self.manifestPath = self.indexDir / "manifest.json"
        self.metadataPath = self.indexDir / "metadata.json"
        self.rawFeaturesPath = self.indexDir / "featuresRaw.npy"
        self.reducedFeaturesPath = self.indexDir / "featuresReduced.npy"

    def loadManifest(self) -> dict:
        if self.manifestPath.exists():
            return json.loads(self.manifestPath.read_text())
        return {"processedVideoIds": [], "collections": []}

    def saveManifest(self, manifest: dict) -> None:
        self.manifestPath.write_text(json.dumps(manifest, indent=2))

    def loadMetadata(self) -> list:
        if self.metadataPath.exists():
            return json.loads(self.metadataPath.read_text())
        return []

    def saveMetadata(self, metadata: list) -> None:
        self.metadataPath.write_text(json.dumps(metadata, indent=2))

    def loadRawFeatures(self):
        if self.rawFeaturesPath.exists():
            return np.load(self.rawFeaturesPath)
        return None

    def saveRawFeatures(self, features: np.ndarray) -> None:
        np.save(self.rawFeaturesPath, features)

    def loadReducedFeatures(self, dimension: int = None):
        path = self.reducedFeaturesPath if dimension is None else self.indexDir / f"featuresReduced_{dimension}.npy"
        if path.exists():
            return np.load(path)
        return None

    def saveReducedFeatures(self, features: np.ndarray, dimension: int = None) -> None:
        path = self.reducedFeaturesPath if dimension is None else self.indexDir / f"featuresReduced_{dimension}.npy"
        np.save(path, features)

    def hasReducedFeatures(self, dimension: int) -> bool:
        return (self.indexDir / f"featuresReduced_{dimension}.npy").exists()

    def hasExistingIndex(self) -> bool:
        return self.rawFeaturesPath.exists() and self.metadataPath.exists()

    def reset(self) -> None:
        for path in (self.manifestPath, self.metadataPath, self.rawFeaturesPath, self.reducedFeaturesPath):
            if path.exists():
                path.unlink()
        for extra in ("scaler.pkl", "pca.pkl"):
            extraPath = self.indexDir / extra
            if extraPath.exists():
                extraPath.unlink()