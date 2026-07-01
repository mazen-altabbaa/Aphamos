from pathlib import Path
import numpy as np
from tqdm import tqdm
from storage.indexStore import IndexStore
from storage.pcaReducer import PcaDimensionReducer, IdentityDimensionReducer
from index.videoProcessor import VideoProcessor
from index.statsReporter import StatsReporter


class IndexBuilder:
    def __init__(self, config, visionEncoder, asrEngine, modalityWeighter):
        self.config = config
        self.indexStore = IndexStore(config.indexDir)
        self.statsReporter = StatsReporter(config.statsDir)
        self.videoProcessor = VideoProcessor(config, visionEncoder, asrEngine, modalityWeighter)

    def buildOrResume(self, videoPaths: list, collectionName: str = "collection"):
        manifest = self.indexStore.loadManifest()
        processedIds = set(manifest["processedVideoIds"])

        rawFeatures = self.indexStore.loadRawFeatures()
        metadata = self.indexStore.loadMetadata()
        rawFeatures = rawFeatures if rawFeatures is not None else np.empty((0, 0))

        newEmbeddings, newMetadata, videoStatsList = [], [], []

        skippedCount = sum(1 for videoPath in videoPaths if Path(videoPath).stem in processedIds)
        remainingPaths = [videoPath for videoPath in videoPaths if Path(videoPath).stem not in processedIds]

        progressBar = tqdm(remainingPaths, desc="Indexing", unit="video")
        if skippedCount:
            progressBar.write(f"Skipping {skippedCount} already-indexed video(s).")

        for videoPath in progressBar:
            videoId = Path(videoPath).stem
            progressBar.set_postfix_str(videoId)
            embeddings, metadataItems, videoStats = self.videoProcessor.process(videoPath)
            if len(embeddings) > 0:
                newEmbeddings.append(embeddings)
                newMetadata.extend(metadataItems)

            videoStatsList.append(videoStats)
            processedIds.add(videoId)
            manifest["processedVideoIds"] = sorted(processedIds)

            metadata.extend(metadataItems)
            if newEmbeddings:
                stacked = np.vstack(newEmbeddings)
                combinedRaw = stacked if rawFeatures.size == 0 else np.vstack([rawFeatures, stacked])
            else:
                combinedRaw = rawFeatures

            self.indexStore.saveRawFeatures(combinedRaw)
            self.indexStore.saveMetadata(metadata)
            self.indexStore.saveManifest(manifest)

        finalRawFeatures = self.indexStore.loadRawFeatures()
        finalMetadata = self.indexStore.loadMetadata()

        pcaDimensions = [512, 256, 128, 64]
        maxDimension = min(pcaDimensions) if finalRawFeatures is not None else 0

        if finalRawFeatures is not None and len(finalRawFeatures) > maxDimension:
            print(f"Fitting PCA for dimensions: {pcaDimensions}")
            for dim in pcaDimensions:
                if finalRawFeatures.shape[1] < dim:
                    print(f"  Skipping PCA-{dim}: raw feature size ({finalRawFeatures.shape[1]}) < {dim}")
                    continue
                reducer = PcaDimensionReducer(dim)
                reducer.fit(finalRawFeatures)
                reducedFeatures = reducer.transform(finalRawFeatures)
                self.indexStore.saveReducedFeatures(reducedFeatures, dimension=dim)
                reducer.save(self.indexStore.indexDir, dimension=dim)
                varExplained = reducer.explainedVarianceRatioSum()
                print(f"  PCA-{dim}: explained variance = {varExplained:.4f}")
            dimensionReducer = PcaDimensionReducer(self.config.pcaDimension)
            dimensionReducer.fit(finalRawFeatures)
        else:
            dimensionReducer = IdentityDimensionReducer()

        manifest["collections"].append(collectionName)
        self.indexStore.saveManifest(manifest)

        if videoStatsList:
            self.statsReporter.reportCollection(
                collectionName, videoStatsList, dimensionReducer, self.config.usePcaReduction
            )

        return finalMetadata, videoStatsList