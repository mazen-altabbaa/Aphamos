from pathlib import Path
import cv2
import numpy as np
from core.thresholdStrategies import buildThresholdStrategy


class FrameSampler:
    def __init__(self, config):
        self.config = config

    def sample(self, videoPath: str):
        videoId = Path(videoPath).stem
        capture = cv2.VideoCapture(str(videoPath))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        minIntervalFrames = int(fps * self.config.minFrameIntervalSec)

        thresholdStrategy = buildThresholdStrategy(self.config)
        previousSignal = None
        frameIndex = 0
        lastCapturedFrame = -minIntervalFrames
        capturedFrames, framePaths, frameIndices = [], [], []

        while len(capturedFrames) < self.config.maxFramesPerVideo:
            success, frame = capture.read()
            if not success:
                break

            signal = self._computeSignal(frame)
            shouldCapture = previousSignal is None

            if not shouldCapture and (frameIndex - lastCapturedFrame) >= minIntervalFrames:
                diff = self._diff(signal, previousSignal)
                thresholdStrategy.update(diff)
                shouldCapture = thresholdStrategy.shouldCapture(diff)

            if shouldCapture:
                framePath = self.config.framesDir / f"{videoId}_frame_{len(capturedFrames):04d}.jpg"
                resized = cv2.resize(frame, (self.config.frameImageSize, self.config.frameImageSize))
                cv2.imwrite(str(framePath), resized)
                capturedFrames.append(frame)
                framePaths.append(str(framePath))
                frameIndices.append(frameIndex)
                lastCapturedFrame = frameIndex

            previousSignal = signal
            frameIndex += 1

        capture.release()
        return capturedFrames, framePaths, frameIndices, thresholdStrategy.history

    def _computeSignal(self, frame):
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.config.useHistogramDiff:
            histogram = cv2.calcHist([grayFrame], [0], None, [32], [0, 256])
            return cv2.normalize(histogram, histogram).flatten()
        return grayFrame.astype(np.float32)

    def _diff(self, signal, previousSignal):
        if self.config.useHistogramDiff:
            return cv2.compareHist(signal, previousSignal, cv2.HISTCMP_CHISQR)
        return float(np.mean(np.abs(signal - previousSignal)))
