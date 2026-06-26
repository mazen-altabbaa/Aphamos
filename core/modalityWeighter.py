class ConstantModalityWeighter:
    def __init__(self, audioWeight: float, frameWeight: float):
        self.audioWeight = audioWeight
        self.frameWeight = frameWeight

    def computeWeights(self, capturedFrameCount: int, videoDurationSec: float, transcript: str):
        return self.audioWeight, self.frameWeight


class DynamicModalityWeighter:
    def __init__(self, baseAudioWeight: float = 1.0, baseFrameWeight: float = 1.0):
        self.baseAudioWeight = baseAudioWeight
        self.baseFrameWeight = baseFrameWeight

    def computeWeights(self, capturedFrameCount: int, videoDurationSec: float, transcript: str):
        safeDuration = max(videoDurationSec, 1e-6)
        sceneChangeRate = capturedFrameCount / safeDuration
        wordCount = len(transcript.split())
        speechDensity = wordCount / safeDuration

        total = sceneChangeRate + speechDensity + 1e-8
        audioWeight = self.baseAudioWeight * (1 + speechDensity / total)
        frameWeight = self.baseFrameWeight * (1 + sceneChangeRate / total)
        return audioWeight, frameWeight


def buildModalityWeighter(config):
    if config.fusionMode == "dynamic":
        return DynamicModalityWeighter(config.constantAudioWeight, config.constantFrameWeight)
    if config.fusionMode == "constant":
        return ConstantModalityWeighter(config.constantAudioWeight, config.constantFrameWeight)
    raise ValueError(f"Unknown fusion mode: {config.fusionMode}")
