import os, cv2, json, pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image