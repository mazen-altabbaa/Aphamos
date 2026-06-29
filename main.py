import os
from pathlib import Path

_modelsDir = str((Path(__file__).parent / "models").resolve())
os.environ["HF_HOME"] = _modelsDir
os.environ["HF_HUB_CACHE"] = str(Path(_modelsDir) / "hub")

import cv2
from config import SystemConfig
from engines.visionEncoder import MobileClipVisionEncoder
from engines.asrEngine import MoonshineAsrEngine
from core.modalityWeighter import buildModalityWeighter
from storage.indexStore import IndexStore
from index.indexBuilder import IndexBuilder
from retrieval.querySearch import QuerySearch
from dataset.datasetLoader import LocalFolderDatasetLoader, CsvFilteredFolderDatasetLoader


def loadVideoPaths(videosDir: str) -> list:
    csvPath = Path("dataset") / "panda70m_filtered_exact_match.csv"
    if csvPath.exists():
        return CsvFilteredFolderDatasetLoader(videosDir, str(csvPath)).listVideoPaths()
    return LocalFolderDatasetLoader(videosDir).listVideoPaths()


def buildEngines(config: SystemConfig):
    visionEncoder = MobileClipVisionEncoder(
        config.visionModelName, config.visionPretrainedTag, config.device
    )
    asrEngine = MoonshineAsrEngine(config.asrModelId, config.device)
    modalityWeighter = buildModalityWeighter(config)
    return visionEncoder, asrEngine, modalityWeighter


def runIndexingMenu(config: SystemConfig, videoPaths: list):
    visionEncoder, asrEngine, modalityWeighter = buildEngines(config)
    indexBuilder = IndexBuilder(config, visionEncoder, asrEngine, modalityWeighter)
    collectionName = input("Name this collection (e.g. batch01): ").strip() or "collection"
    metadata, videoStatsList = indexBuilder.buildOrResume(videoPaths, collectionName)
    print(f"Indexed {len(videoStatsList)} new videos. Total items in index: {len(metadata)}")
    return visionEncoder


def runQueryMenu(config: SystemConfig, visionEncoder=None):
    if visionEncoder is None:
        visionEncoder = MobileClipVisionEncoder(
            config.visionModelName, config.visionPretrainedTag, config.device
        )
    querySearch = QuerySearch(config, visionEncoder)

    while True:
        print("\n1. Text query  2. Image query  3. Back")
        choice = input("Choice: ").strip()

        if choice not in ("1", "2"):
            return

        print("Retrieval mode: 1. Both  2. Audio only  3. Image only")
        modeChoice = input("Choice: ").strip()
        retrievalMode = {"1": "both", "2": "audio", "3": "image"}.get(modeChoice, "both")

        if choice == "1":
            textQuery = input("Enter search text: ").strip()
            results, timings = querySearch.queryWithText(textQuery, topK=5, retrievalMode=retrievalMode)
        else:
            imagePath = input("Enter image path: ").strip()
            frame = cv2.imread(imagePath)
            results, timings = querySearch.queryWithImage(
                frame, topK=5, retrievalMode=retrievalMode, queryLabel=imagePath
            )

        for rank, result in enumerate(results, start=1):
            print(f"{rank}. {result['videoId']} | score={result['score']:.4f}")
        print("Timings (sec):", {key: round(value, 4) for key, value in timings.items()})


def main():
    config = SystemConfig()
    config.ensureDirectories()
    indexStore = IndexStore(config.indexDir)

    if not indexStore.hasExistingIndex():
        videosDir = input("Videos directory: ").strip()
        videoPaths = loadVideoPaths(videosDir)
        if not videoPaths:
            print("No videos found.")
            return
        visionEncoder = runIndexingMenu(config, videoPaths)
        runQueryMenu(config, visionEncoder)
        return

    print("1. Use existing index\n2. Add more videos\n3. Reset index from scratch")
    choice = input("Choice: ").strip()

    if choice == "2":
        videosDir = input("Videos directory: ").strip()
        videoPaths = loadVideoPaths(videosDir)
        visionEncoder = runIndexingMenu(config, videoPaths)
        runQueryMenu(config, visionEncoder)
    elif choice == "3":
        indexStore.reset()
        videosDir = input("Videos directory: ").strip()
        videoPaths = loadVideoPaths(videosDir)
        visionEncoder = runIndexingMenu(config, videoPaths)
        runQueryMenu(config, visionEncoder)
    else:
        runQueryMenu(config)


if __name__ == "__main__":
    main()