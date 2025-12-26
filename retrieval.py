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