import time
from pathlib import Path
import cv2
import numpy as np
from core.frameSampler import FrameSampler


class VideoProcessor:
    def __init__(self, config, visionEncoder, asrEngine, modalityWeighter):
        self.config = config
        self.visionEncoder = visionEncoder
        self.asrEngine = asrEngine
        self.modalityWeighter = modalityWeighter
        self.frameSampler = FrameSampler(config)

    def process(self, videoPath: str):
        startTime = time.perf_counter()
        videoId = Path(videoPath).stem
        capture = cv2.VideoCapture(str(videoPath))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frameCount = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        durationSec = frameCount / fps if fps > 0 else 0.0
        capture.release()

        frames, framePaths, frameIndices, thresholdHistory = self.frameSampler.sample(videoPath)
        transcript = self.asrEngine.transcribe(videoPath)

        audioWeight, frameWeight = self.modalityWeighter.computeWeights(
            thresholdHistory, transcript
        )

        embeddings, metadataItems = [], []
        for frame, framePath, frameIndex in zip(frames, framePaths, frameIndices):
            embeddings.append(self.visionEncoder.encodeImage(frame))
            metadataItems.append({
                "videoId": videoId,
                "videoPath": str(videoPath),
                "modality": "frame",
                "frameIndex": frameIndex,
                "framePath": framePath,
                "appliedWeight": frameWeight,
            })

        if transcript:
            embeddings.append(self.visionEncoder.encodeText(transcript))
            metadataItems.append({
                "videoId": videoId,
                "videoPath": str(videoPath),
                "modality": "transcript",
                "transcript": transcript,
                "appliedWeight": audioWeight,
            })

        elapsedSec = time.perf_counter() - startTime
        videoStats = {
            "videoId": videoId,
            "durationSec": durationSec,
            "capturedFrameCount": len(frames),
            "processingTimeSec": elapsedSec,
            "thresholdHistory": thresholdHistory,
            "audioWeight": audioWeight,
            "frameWeight": frameWeight,
            "transcriptWordCount": len(transcript.split()),
        }

        if not embeddings:
            return np.empty((0, self.visionEncoder.embeddingDim)), [], videoStats
        return np.array(embeddings), metadataItems, videoStats
