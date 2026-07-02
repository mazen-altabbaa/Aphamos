from dataclasses import dataclass
from pathlib import Path


@dataclass
class SystemConfig:
    outputDir: str = "output"
    modelsDir: str = "models"
    frameImageSize: int = 224
    maxFramesPerVideo: int = 30
    minFrameIntervalSec: float = 2.0

    thresholdMode: str = "adaptive"
    initialThreshold: float = 15.0
    thresholdLearningRate: float = 0.1
    thresholdHistoryWindow: int = 9
    constantThresholdValue: float = 70.0
    randomThresholdMin: float = 15.0
    randomThresholdMax: float = 100.0
    useHistogramDiff: bool = True

    usePcaReduction: bool = True
    pcaDimension: int = 256

    fusionMode: str = "dynamic"
    constantAudioWeight: float = 1.5
    constantFrameWeight: float = 1.0
    retrievalMode: str = "both"

    asrModelId: str = "UsefulSensors/moonshine-base"
    visionModelName: str = "MobileCLIP-S2"
    visionPretrainedTag: str = "datacompdr"
    device: str = "cuda"

    @property
    def framesDir(self) -> Path:
        return Path(self.outputDir) / "frames"

    @property
    def indexDir(self) -> Path:
        return Path(self.outputDir) / "index"

    @property
    def statsDir(self) -> Path:
        return Path(self.outputDir) / "stats"

    def ensureDirectories(self) -> None:
        for path in (self.framesDir, self.indexDir, self.statsDir, Path(self.modelsDir)):
            path.mkdir(parents=True, exist_ok=True)
