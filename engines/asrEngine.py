import re
import warnings
import numpy as np
import torch
import librosa
from transformers import AutoProcessor, MoonshineForConditionalGeneration
from core.interfaces import IAsrEngine

SAMPLE_RATE = 16000
CHUNK_SEC = 30
NUM_CHUNKS = 4


class MoonshineAsrEngine(IAsrEngine):
    def __init__(self, modelId: str, device: str):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(modelId)
        self.model = MoonshineForConditionalGeneration.from_pretrained(modelId).to(device)
        self.model.eval()

    def _videoDurationSec(self, videoPath: str) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            duration = librosa.get_duration(path=str(videoPath))
        return duration

    def _loadChunk(self, videoPath: str, offsetSec: float) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, _ = librosa.load(str(videoPath), sr=SAMPLE_RATE, mono=True,
                                    offset=offsetSec, duration=CHUNK_SEC)
        return audio

    def _chunkOffsets(self, durationSec: float) -> list:
        if durationSec <= CHUNK_SEC:
            return [0.0]
        usable = durationSec - CHUNK_SEC
        return [usable * i / (NUM_CHUNKS - 1) for i in range(NUM_CHUNKS)]

    def _removeRepetitions(self, text: str) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        seen = []
        for sentence in sentences:
            if sentence not in seen:
                seen.append(sentence)
        return " ".join(seen)

    @torch.no_grad()
    def _transcribeAudio(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) < SAMPLE_RATE:
            return ""
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE,
                                return_tensors="pt").to(self.device)
        generatedIds = self.model.generate(
            **inputs,
            max_new_tokens=128,
            repetition_penalty=1.3,
            no_repeat_ngram_size=6,
        )
        text = self.processor.decode(generatedIds[0], skip_special_tokens=True)
        return self._removeRepetitions(text.strip())

    def transcribe(self, videoPath: str) -> str:
        durationSec = self._videoDurationSec(str(videoPath))
        offsets = self._chunkOffsets(durationSec)
        parts = []
        for offset in offsets:
            audio = self._loadChunk(str(videoPath), offset)
            text = self._transcribeAudio(audio)
            if text and text not in parts:
                parts.append(text)
        return " ".join(parts)