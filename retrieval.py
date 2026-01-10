import torch
import whisper
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import os, cv2, json, pickle
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

def checkCuda():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128' 


    print("GPU CONFIGURATION CHECK")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        for i in range(device_count):
            gpuName = torch.cuda.get_device_name(i)
            gpuMemory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpuName} ({gpuMemory:.1f} GB)")
        
        torch.cuda.set_device(0)
        currDevice = torch.cuda.current_device()
        print(f"\nCurrent GPU: {torch.cuda.get_device_name(currDevice)}")
        
        cudnn.benchmark = True 
        cudnn.enabled = True
        
        testTensor = torch.randn(1000, 1000).cuda()
        print(f"GPU test passed: {testTensor.shape} on GPU")
        del testTensor
        torch.cuda.empty_cache()
    else:
        print("WARNING: No CUDA GPU detected!")
        print("Check if NVIDIA drivers are installed properly")


    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using device: {torch.cuda.get_device_name(device)}")
        
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        print("Using CPU")


def checkFFMPEG():
    import subprocess
    
    print("Test - CHECKING FFMPEG")
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        print(f"ffmpeg available: {result.returncode == 0}")
        if result.returncode == 0:
            print(f"ffmpeg version: {result.stdout.split('version')[1].split()[0]}")
    except FileNotFoundError:
        print("ffmpeg NOT FOUND in PATH")
    
    try:
        result = subprocess.run(['ffprobe', '-version'], capture_output=True, text=True)
        print(f"ffprobe available: {result.returncode == 0}")
    except FileNotFoundError:
        print("ffprobe NOT FOUND in PATH")
    


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using device: {torch.cuda.get_device_name(device)}")
    
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    print("Using CPU")

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleared. Current usage: {torch.cuda.memory_allocated()/1024**2:.2f} MB")


whisperModel = None

class Settings:
    videosDir = "Dataset"
    outputDir = "output"
    imgSize = 224
    frameRate = 12
    maxFrames = 100
    featureDim = 14

    Path(outputDir).mkdir(exist_ok=True)
    Path(f"{outputDir}/frames").mkdir(exist_ok=True)
    Path(f"{outputDir}/index").mkdir(exist_ok=True)


clipModel, clipProcessor = None, None
localModelPath = "D:/temp/curr/clip-vit-base-patch32"
localWhisperPath = "D:/temp/curr/whisper-small" 

if os.path.exists(localModelPath):
    print("Using local CLIP model from:", localModelPath)
    try:
        clipModel = CLIPModel.from_pretrained(localModelPath, local_files_only=True).to(device)
        clipProcessor = CLIPProcessor.from_pretrained(localModelPath, local_files_only=True)
    except:
        print("Failed to load local CLIP model, downloading...")
        clipModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clipProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
else:
    print("Local CLIP model not found, downloading from Hugging Face...")
    clipModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clipProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

if torch.cuda.is_available():
    clipModel = clipModel.to("cuda:0") 
    print(f"CLIP model moved to GPU 0: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CLIP running on CPU - very slow!")



whisper_model_path = os.path.join(localWhisperPath, "model.pt")
if os.path.exists(whisper_model_path):
    print("Using local Whisper model from:", whisper_model_path)
    whisperModel = whisper.load_model(whisper_model_path).to(device)
else:
    print(f"Local Whisper model not found at {whisper_model_path}")
    print("Attempting to load from Hugging Face...")
    
    if os.path.isdir(localWhisperPath):
        try:
            whisperModel = whisper.load_model("small", download_root=localWhisperPath).to(device)
            print(f"Model downloaded and saved to {localWhisperPath}")
        except:
            print("Download failed, trying without download root...")
            whisperModel = whisper.load_model("small").to(device)
    else:
        whisperModel = whisper.load_model("small").to(device)
        print(f"Saving model to {localWhisperPath} for future use...")
        os.makedirs(os.path.dirname(localWhisperPath), exist_ok=True)

if torch.cuda.is_available():
    whisperModel = whisperModel.to(device)
    print("Whisper model moved to GPU")
checkCuda()


@torch.no_grad()
def extractVideoEmbeddings(videoPath, maxFrames=Settings.maxFrames, minIntervalSec=2, 
                          initialThreshold=25, learningRate=0.1):
    cap = cv2.VideoCapture(str(videoPath))
    embeddings, captureTimes, framePaths = [], [], []
    frameCount, savedCount = 0, 0
    videoId = Path(videoPath).stem
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
        
    minIntervalFrames = int(fps * minIntervalSec)
    
    prevFrame = None
    prevHist = None
    lastCapturedFrame = -minIntervalFrames
    
    currentThreshold = initialThreshold
    diffHistory = []
    
    while savedCount < maxFrames:
        ret, frame = cap.read()
        if not ret:
            break
            
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([grayFrame], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        shouldCapture = False
        
        if prevHist is None:
            shouldCapture = True
        elif (frameCount - lastCapturedFrame) >= minIntervalFrames:
            diff = cv2.compareHist(hist, prevHist, cv2.HISTCMP_CHISQR)
            diffHistory.append(diff)
            
            if len(diffHistory) > 9:
                avgDiff = np.mean(diffHistory[-9:])
                currentThreshold = initialThreshold * (1 + learningRate * (avgDiff - initialThreshold) / initialThreshold)    
            if diff > currentThreshold:
                shouldCapture = True
        
        if shouldCapture:
            embedding = getImageEmbedding(frame)
            embeddings.append(embedding)
            captureTimes.append(frameCount)
            
            framePath = f"{Settings.outputDir}/frames/{videoId}_frame_{savedCount:04d}.jpg"
            cv2.imwrite(framePath, cv2.resize(frame, (Settings.imgSize, Settings.imgSize)))
            framePaths.append(framePath)
            savedCount += 1
            lastCapturedFrame = frameCount
        
        prevFrame = frame
        prevHist = hist
        frameCount += 1
    
    cap.release()
    
    metadata = [
        {
            "videoId": videoId,
            "videoPath": str(videoPath),
            "frameIndex": time,
            "framePath": path
        }
        for time, path in zip(captureTimes, framePaths)
    ]
    
    print(f"Extracted {savedCount} frames from {videoId} with adaptive threshold {currentThreshold:.2f}")
    
    return np.array(embeddings), metadata


def getImageEmbedding(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    inputs = clipProcessor(images=img, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        emb = clipModel.get_image_features(**inputs)
    return emb.cpu().numpy().flatten()


def getTextEmbedding(text):
    inputs = clipProcessor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        emb = clipModel.get_text_features(**inputs)
    return emb.cpu().numpy().flatten()


def extractTranscript(videoPath):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    result = whisperModel.transcribe(
        str(videoPath),
        fp16=torch.cuda.is_available()
    )
    return result["text"]



def getTranscriptEmbedding(videoPath):
    try:
        transcript = extractTranscript(videoPath)
        emb = getTextEmbedding(transcript)
        return emb, transcript
    except Exception as e:
        print(f"Error extracting transcript for {videoPath}: {e}")
        emb = np.zeros(clipModel.config.projection_dim)
        return emb, ""

def processVideos(videoFiles, skip=False):
    processed_videos = set()
    if skip:
        try:
            with open(f"{Settings.outputDir}/index/metadata.json") as f:
                existing_meta = json.load(f)
                processed_videos = {m['videoId'] for m in existing_meta
                                  if m.get('videoId') and not m.get('transcript')}
                print(f"Found {len(processed_videos)} already processed videos")
        except:
            pass

    allEmbeddings, allMeta = [], []

    for vp in tqdm(videoFiles, desc="Processing videos"):
        vidId = Path(vp).stem

        if skip and vidId in processed_videos:
            print(f"Skipping already processed video: {vidId}")
            continue

        embFrames, metaFrames = extractVideoEmbeddings(vp)
        allEmbeddings.extend(embFrames)
        allMeta.extend(metaFrames)

        embTranscript, transcript = getTranscriptEmbedding(vp)
        allEmbeddings.append(embTranscript)
        allMeta.append({
            "videoId": vidId,
            "videoPath": str(vp),
            "frameIndex": None,
            "framePath": None,
            "transcript": transcript
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


    return np.array(allEmbeddings), allMeta


def buildIndex(features, metadata, incremental=False):
    if incremental:
        existing = loadExistingIndex()

        if existing['features'] is not None and existing['metadata'] is not None:
            print(f"Updating existing index with {len(features)} new items")

            if existing['scaler'] is not None and existing['pca'] is not None:
                print("Transforming new features using existing scaler and PCA...")

                featuresNorm = existing['scaler'].transform(features)
                featuresRed = existing['pca'].transform(featuresNorm)
                allFeaturesRed = np.vstack([existing['features'], featuresRed])

                allFeaturesOriginal = None

                if os.path.exists(f"{Settings.outputDir}/index/features_original.npy"):
                    existingOriginal = np.load(f"{Settings.outputDir}/index/features_original.npy")
                    allFeaturesOriginal = np.vstack([existingOriginal, features])
                else:
                    print("Warning: Original features not found. Saving new ones only.")
                    allFeaturesOriginal = features

                np.save(f"{Settings.outputDir}/index/features_original.npy", allFeaturesOriginal)

            else:
                print("Error: Existing scaler or PCA not found!")
                print("Cannot do incremental update without them.")
                print("Falling back to rebuilding from scratch...")
                incremental = False
                allFeaturesRed = None
                allFeaturesOriginal = None

            allMetadata = existing['metadata'] + metadata

        else:
            print("No existing index found, creating new one")
            incremental = False
            allFeaturesRed = None
            allMetadata = metadata
            allFeaturesOriginal = features
    else:
        print("Building new index (overwriting existing)")
        allFeaturesRed = None
        allMetadata = metadata
        allFeaturesOriginal = features

    if not incremental or allFeaturesRed is None:
        print("Fitting new scaler and PCA...")
        scaler = StandardScaler()
        allFeaturesNorm = scaler.fit_transform(allFeaturesOriginal)

        pca = PCA(n_components=Settings.featureDim)
        allFeaturesRed = pca.fit_transform(allFeaturesNorm)

        np.save(f"{Settings.outputDir}/index/features_original.npy", allFeaturesOriginal)
    else:
        scaler = existing['scaler']
        pca = existing['pca']

    np.save(f"{Settings.outputDir}/index/features.npy", allFeaturesRed)
    with open(f"{Settings.outputDir}/index/metadata.json", "w") as f:
        json.dump(allMetadata, f, indent=2)

    with open(f"{Settings.outputDir}/index/pca.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open(f"{Settings.outputDir}/index/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"Index {'updated' if incremental else 'built'} with shape: {allFeaturesRed.shape}")
    print(f"Total metadata items: {len(allMetadata)}")

    return allFeaturesRed, allMetadata


def loadExistingIndex():
    indexDir = f"{Settings.outputDir}/index"
    existingData = {
        'features': None,
        'metadata': None,
        'scaler': None,
        'pca': None,
        'features_original': None
    }

    try:
        if os.path.exists(f"{indexDir}/features.npy"):
            existingData['features'] = np.load(f"{indexDir}/features.npy")
            print(f"Loaded existing features: {existingData['features'].shape}")

        if os.path.exists(f"{indexDir}/features_original.npy"):
            existingData['features_original'] = np.load(f"{indexDir}/features_original.npy")
            print(f"Loaded original features: {existingData['features_original'].shape}")

        if os.path.exists(f"{indexDir}/metadata.json"):
            with open(f"{indexDir}/metadata.json") as f:
                existingData['metadata'] = json.load(f)
            print(f"Loaded existing metadata: {len(existingData['metadata'])} items")

        if os.path.exists(f"{indexDir}/scaler.pkl"):
            with open(f"{indexDir}/scaler.pkl", "rb") as f:
                existingData['scaler'] = pickle.load(f)
            print("Loaded existing scaler")

        if os.path.exists(f"{indexDir}/pca.pkl"):
            with open(f"{indexDir}/pca.pkl", "rb") as f:
                existingData['pca'] = pickle.load(f)
            print("Loaded existing PCA")

    except Exception as e:
        print(f"Error loading existing index: {e}")
        existingData = {
            'features': None,
            'metadata': None,
            'scaler': None,
            'pca': None,
            'features_original': None
        }

    return existingData


def loadPCAandScaler(outputDir=Settings.outputDir):
    with open(f"{outputDir}/index/pca.pkl","rb") as f:
        pca = pickle.load(f)
    with open(f"{outputDir}/index/scaler.pkl","rb") as f:
        scaler = pickle.load(f)
    return pca, scaler

def getTranscriptEmbedding(videoPath):
    try:
        transcript = extractTranscript(videoPath)
        emb = getTextEmbedding(transcript)
        return emb, transcript
    except Exception as e:
        print(f"Error extracting transcript for {videoPath}: {e}")
        emb = np.zeros(clipModel.config.projection_dim)
        return emb, ""



def loadExistingIndex():
    indexDir = f"{Settings.outputDir}/index"
    existingData = {
        'features': None,
        'metadata': None,
        'scaler': None,
        'pca': None,
        'features_original': None
    }

    try:
        if os.path.exists(f"{indexDir}/features.npy"):
            existingData['features'] = np.load(f"{indexDir}/features.npy")
            print(f"Loaded existing features: {existingData['features'].shape}")

        if os.path.exists(f"{indexDir}/features_original.npy"):
            existingData['features_original'] = np.load(f"{indexDir}/features_original.npy")
            print(f"Loaded original features: {existingData['features_original'].shape}")

        if os.path.exists(f"{indexDir}/metadata.json"):
            with open(f"{indexDir}/metadata.json") as f:
                existingData['metadata'] = json.load(f)
            print(f"Loaded existing metadata: {len(existingData['metadata'])} items")

        if os.path.exists(f"{indexDir}/scaler.pkl"):
            with open(f"{indexDir}/scaler.pkl", "rb") as f:
                existingData['scaler'] = pickle.load(f)
            print("Loaded existing scaler")

        if os.path.exists(f"{indexDir}/pca.pkl"):
            with open(f"{indexDir}/pca.pkl", "rb") as f:
                existingData['pca'] = pickle.load(f)
            print("Loaded existing PCA")

    except Exception as e:
        print(f"Error loading existing index: {e}")
        existingData = {
            'features': None,
            'metadata': None,
            'scaler': None,
            'pca': None,
            'features_original': None
        }

    return existingData

def transformQuery(qRaw, outputDir=Settings.outputDir):
    pca, scaler = loadPCAandScaler(outputDir)
    qNorm = scaler.transform([qRaw])
    qRed = pca.transform(qNorm)
    return qRed.flatten()

def extractTranscript(videoPath):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    result = whisperModel.transcribe(
        str(videoPath),
        fp16=torch.cuda.is_available()
    )
    return result["text"]


def queryWithImage(imgPath, topK=5):
    feats = np.load(f"{Settings.outputDir}/index/features.npy")
    with open(f"{Settings.outputDir}/index/metadata.json") as f: meta = json.load(f)

    frame = cv2.imread(imgPath)
    qRaw = getImageEmbedding(frame)
    q = transformQuery(qRaw.flatten())

    top, sims = search(q, feats, meta, topK)
    print("Image Query:", imgPath)
    for r,(i,s) in enumerate(zip(top,sims),1):
        m = meta[i]
        print(f"{r}. {m['videoId']} | Frame {m['frameIndex']} | {m['framePath']} | Sim={s:.3f}")



def queryWithText(textQuery, topK=5):
    feats = np.load(f"{Settings.outputDir}/index/features.npy")
    with open(f"{Settings.outputDir}/index/metadata.json") as f:
        meta = json.load(f)

    qRaw = getTextEmbedding(textQuery)
    q = transformQuery(qRaw.flatten())

    top, sims = search(q, feats, meta, topK)
    print("Text Query:", textQuery)
    for r,(i,s) in enumerate(zip(top,sims),1):
        m = meta[i]
        if m.get("transcript"):
            print(f"{r}. {m['videoId']} | Transcript match | Sim={s:.3f}")
            print("   Transcript snippet:", m['transcript'][:120], "...")
        else:
            print(f"{r}. {m['videoId']} | Frame {m['frameIndex']} | {m['framePath']} | Sim={s:.3f}")


def search(queryFeat, indexFeats, metadata, topK=5, weight_transcript=1.5, weight_frame=1.0):
    q = queryFeat / (np.linalg.norm(queryFeat) + 1e-10)
    idxNorm = indexFeats / (np.linalg.norm(indexFeats, axis=1, keepdims=True) + 1e-10)
    sims = np.dot(idxNorm, q)

    weighted_sims = []
    for i, sim in enumerate(sims):
        m = metadata[i]
        if m.get("transcript"):
            weighted_sims.append(sim * weight_transcript)
        else:
            weighted_sims.append(sim * weight_frame)

    weighted_sims = np.array(weighted_sims)
    top = np.argsort(weighted_sims)[::-1][:topK]
    return top, weighted_sims[top]


def main(incremental=False, skip=True):
    exts = [".mp4", ".avi", ".mov", ".mkv", ".flv"]
    vids = [p for e in exts for p in Path(Settings.videosDir).glob(f"*{e}")]

    if not vids:
        print("No videos found.")
        return None, None

    if incremental and not os.path.exists(f"{Settings.outputDir}/index/features.npy"):
        print("No existing index found, starting fresh")
        incremental = False

    feats, meta = processVideos(vids, skip=skip)

    if len(feats) > 0:
        return buildIndex(feats, meta, incremental=incremental)
    else:
        print("No new videos to process")
        return None, None


def ShowChoices():

    while True:
        print("\nOptions:")
        print("1. Search with text query: ")
        print("2. Exit")

        choice = input("\nEnter choice (1-2): ")

        if choice == "1":
            query = input("Enter search query: ")
            queryWithText(query, topK=5)

        elif choice == "2":
            print("Goodbye!")
            break

        else:
            print("please enter 1-3!!!")


if __name__ == "__main__":
    checkFFMPEG()

    indexfound = os.path.exists(f"{Settings.outputDir}/index/features.npy")

    if not indexfound:
        print("No index found. Building initial index...")
        main(incremental=False, skip=False)
    else:
        print("Index already exists!")
        print("\nOptions:")
        print("1. Use existing index")
        print("2. Add new videos to index (incremental update)")
        print("3. Rebuild index from scratch (DELETE existing)")

        choice = input("\nEnter choice (1-3): ")

        if choice == "1":
            print("Using existing index...")
        elif choice == "2": 
            print("Adding new videos incrementally...")
            main(incremental=True, skip=True)
        elif choice == "3":
            import shutil
            if os.path.exists(Settings.outputDir):
                shutil.rmtree(Settings.outputDir)
            Path(Settings.outputDir).mkdir(exist_ok=True)
            Path(f"{Settings.outputDir}/frames").mkdir(exist_ok=True)
            Path(f"{Settings.outputDir}/index").mkdir(exist_ok=True)

            print("Building fresh index...")
            main(incremental=False, skip=False)
        else:
            print("Invalid choice. Using existing index...")

    print("\n" + "=" * 60)
    print("Starting enhanced system with Rocchio feedback...")
    ShowChoices()