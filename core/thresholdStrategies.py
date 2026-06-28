import random
from core.interfaces import IThresholdStrategy


class AdaptiveThresholdStrategy(IThresholdStrategy):
    def __init__(self, initialThreshold: float, learningRate: float, historyWindow: int):
        self.initialThreshold = initialThreshold
        self.learningRate = learningRate
        self.historyWindow = historyWindow
        self.diffHistory = []
        self.thresholdHistory = []
        self.currentThreshold = initialThreshold

    def update(self, diff: float) -> float:
        self.diffHistory.append(diff)
        recentDiffs = self.diffHistory[-self.historyWindow:]
        meanRecent = sum(recentDiffs) / len(recentDiffs)
        eta = self.learningRate
        self.currentThreshold = self.initialThreshold * (1 - eta) + eta * meanRecent
        self.thresholdHistory.append(self.currentThreshold)
        return self.currentThreshold

    def shouldCapture(self, diff: float) -> bool:
        return diff > self.currentThreshold

    @property
    def history(self) -> list:
        return self.thresholdHistory


class ConstantThresholdStrategy(IThresholdStrategy):
    def __init__(self, thresholdValue: float):
        self.thresholdValue = thresholdValue
        self.thresholdHistory = []

    def update(self, diff: float) -> float:
        self.thresholdHistory.append(self.thresholdValue)
        return self.thresholdValue

    def shouldCapture(self, diff: float) -> bool:
        return diff > self.thresholdValue

    @property
    def history(self) -> list:
        return self.thresholdHistory


class RandomThresholdStrategy(IThresholdStrategy):
    def __init__(self, minValue: float, maxValue: float):
        self.minValue = minValue
        self.maxValue = maxValue
        self.thresholdHistory = []
        self.currentThreshold = random.uniform(minValue, maxValue)

    def update(self, diff: float) -> float:
        self.currentThreshold = random.uniform(self.minValue, self.maxValue)
        self.thresholdHistory.append(self.currentThreshold)
        return self.currentThreshold

    def shouldCapture(self, diff: float) -> bool:
        return diff > self.currentThreshold

    @property
    def history(self) -> list:
        return self.thresholdHistory


def buildThresholdStrategy(config) -> IThresholdStrategy:
    if config.thresholdMode == "adaptive":
        return AdaptiveThresholdStrategy(
            config.initialThreshold,
            config.thresholdLearningRate,
            config.thresholdHistoryWindow,
        )
    if config.thresholdMode == "constant":
        return ConstantThresholdStrategy(config.constantThresholdValue)
    if config.thresholdMode == "random":
        return RandomThresholdStrategy(config.randomThresholdMin, config.randomThresholdMax)
    raise ValueError(f"Unknown threshold mode: {config.thresholdMode}")
