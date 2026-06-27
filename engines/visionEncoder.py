import cv2
import torch
import open_clip
from PIL import Image
from core.interfaces import IVisionEncoder


class MobileClipVisionEncoder(IVisionEncoder):
    def __init__(self, modelName: str, pretrainedTag: str, device: str):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            modelName, pretrained=pretrainedTag
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(modelName)
        self._embeddingDim = self.model.text_projection.shape[1] if hasattr(self.model, "text_projection") else 512

    @torch.no_grad()
    def encodeImage(self, frame):
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(rgbFrame)
        tensor = self.preprocess(pilImage).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        return features.cpu().numpy().flatten()

    @torch.no_grad()
    def encodeText(self, text: str):
        tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(tokens)
        return features.cpu().numpy().flatten()

    @property
    def embeddingDim(self) -> int:
        return self._embeddingDim
