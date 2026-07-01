import pickle
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from core.interfaces import IDimensionReducer


class PcaDimensionReducer(IDimensionReducer):
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=dimension)
        self._fitted = False

    def fit(self, features) -> None:
        normalized = self.scaler.fit_transform(features)
        self.pca.fit(normalized)
        self._fitted = True

    def transform(self, features):
        normalized = self.scaler.transform(features)
        return self.pca.transform(normalized)

    def isFitted(self) -> bool:
        return self._fitted

    def explainedVarianceRatioSum(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum()) if self._fitted else 0.0

    def save(self, indexDir: Path, dimension: int = None) -> None:
        suffix = f"_{dimension}" if dimension is not None else ""
        with open(indexDir / f"scaler{suffix}.pkl", "wb") as scalerFile:
            pickle.dump(self.scaler, scalerFile)
        with open(indexDir / f"pca{suffix}.pkl", "wb") as pcaFile:
            pickle.dump(self.pca, pcaFile)

    @classmethod
    def load(cls, indexDir: Path, dimension: int):
        reducer = cls(dimension)
        scalerPath = indexDir / f"scaler_{dimension}.pkl"
        pcaPath = indexDir / f"pca_{dimension}.pkl"
        if not scalerPath.exists():
            scalerPath = indexDir / "scaler.pkl"
            pcaPath = indexDir / "pca.pkl"
        if scalerPath.exists() and pcaPath.exists():
            with open(scalerPath, "rb") as scalerFile:
                reducer.scaler = pickle.load(scalerFile)
            with open(pcaPath, "rb") as pcaFile:
                reducer.pca = pickle.load(pcaFile)
            reducer._fitted = True
        return reducer


class IdentityDimensionReducer(IDimensionReducer):
    def fit(self, features) -> None:
        pass

    def transform(self, features):
        return features

    def isFitted(self) -> bool:
        return True

    def save(self, indexDir: Path, dimension: int = None) -> None:
        pass

    @classmethod
    def load(cls, indexDir: Path, dimension: int):
        return cls()