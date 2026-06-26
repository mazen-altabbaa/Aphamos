from abc import ABC, abstractmethod
import numpy as np


class IAsrEngine(ABC):
    @abstractmethod
    def transcribe(self, videoPath: str) -> str:
        ...


class IVisionEncoder(ABC):
    @abstractmethod
    def encodeImage(self, frame: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def encodeText(self, text: str) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def embeddingDim(self) -> int:
        ...


class IThresholdStrategy(ABC):
    @abstractmethod
    def shouldCapture(self, diff: float) -> bool:
        ...

    @abstractmethod
    def update(self, diff: float) -> float:
        ...

    @property
    @abstractmethod
    def history(self) -> list:
        ...


class IDimensionReducer(ABC):
    @abstractmethod
    def fit(self, features: np.ndarray) -> None:
        ...

    @abstractmethod
    def transform(self, features: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def isFitted(self) -> bool:
        ...
