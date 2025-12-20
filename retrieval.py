import torch
import whisper
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import os, cv2, json, pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity



clipModel, clipProcessor = None, None
localModelPath = "models/clip-vit-base-patch32"
localWhisperPath = "models/whisper-small"

if os.path.exists(localModelPath):
    print("Using local CLIP model from:", localModelPath)
    try:
        clipModel = CLIPModel.from_pretrained(localModelPath, local_files_only=True)
        clipProcessor = CLIPProcessor.from_pretrained(localModelPath, local_files_only=True)
    except:
        print("Failed to load local CLIP model, downloading...")
        clipModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clipProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
else:
    print("Local CLIP model not found, downloading from Hugging Face...")
    clipModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clipProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

whisper_model_path = os.path.join(localWhisperPath, "model.pt")
if os.path.exists(whisper_model_path):
    print("Using local Whisper model from:", whisper_model_path)
    whisperModel = whisper.load_model(whisper_model_path)
else:
    print(f"Local Whisper model not found at {whisper_model_path}")
    print("Attempting to load from Hugging Face...")
    
    if os.path.isdir(localWhisperPath):
        try:
            whisperModel = whisper.load_model("small", download_root=localWhisperPath)
            print(f"Model downloaded and saved to {localWhisperPath}")
        except:
            print("Download failed, trying without download root...")
            whisperModel = whisper.load_model("small")
    else:
        whisperModel = whisper.load_model("small")
        print(f"Saving model to {localWhisperPath} for future use...")
        os.makedirs(os.path.dirname(localWhisperPath), exist_ok=True)
