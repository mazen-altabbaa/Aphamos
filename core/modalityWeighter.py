import math
from collections import Counter


class ConstantModalityWeighter:
    def __init__(self, audioWeight: float, frameWeight: float):
        self.audioWeight = audioWeight
        self.frameWeight = frameWeight

    def computeWeights(self, frameDiffHistory: list, transcript: str):
        return self.audioWeight, self.frameWeight


class EntropyModalityWeighter:
    def computeWeights(self, frameDiffHistory: list, transcript: str):
        wordCounts = list(Counter(transcript.split()).values())
        frameEntropy = self._shannonEntropy(frameDiffHistory)
        audioEntropy = self._shannonEntropy(wordCounts)

        frameTrust = 1.0 - frameEntropy
        audioTrust = 1.0 - audioEntropy
        totalTrust = frameTrust + audioTrust

        if totalTrust <= 1e-12:
            return 1.0, 1.0

        frameWeight = frameTrust / totalTrust
        audioWeight = audioTrust / totalTrust
        return audioWeight, frameWeight

    def _shannonEntropy(self, values: list) -> float:
        sampleCount = len(values)
        if sampleCount <= 1:
            return 0.0

        total = sum(values)
        if total <= 1e-12:
            return 0.0

        probabilities = [value / total for value in values if value > 0]
        rawEntropy = -sum(probability * math.log(probability) for probability in probabilities)
        return rawEntropy / math.log(sampleCount)


def buildModalityWeighter(config):
    if config.fusionMode == "dynamic":
        return EntropyModalityWeighter()
    if config.fusionMode == "constant":
        return ConstantModalityWeighter(config.constantAudioWeight, config.constantFrameWeight)
    raise ValueError(f"Unknown fusion mode: {config.fusionMode}")
