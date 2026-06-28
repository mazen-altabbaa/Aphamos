import librosa
import torch
from transformers import AutoProcessor, MoonshineForConditionalGeneration
from core.interfaces import IAsrEngine


class MoonshineAsrEngine(IAsrEngine):
    def __init__(self, modelId: str, device: str):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(modelId)
        self.model = MoonshineForConditionalGeneration.from_pretrained(modelId).to(device)
        self.model.eval()

    @torch.no_grad()
    def transcribe(self, videoPath: str) -> str:
        audio, sampleRate = librosa.load(str(videoPath), sr=16000)
        inputs = self.processor(audio, sampling_rate=sampleRate, return_tensors="pt").to(self.device)
        generatedIds = self.model.generate(**inputs, max_new_tokens=256)
        text = self.processor.decode(generatedIds[0], skip_special_tokens=True)
        return text.strip()
