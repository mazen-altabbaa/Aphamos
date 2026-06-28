from pathlib import Path
import numpy as np
from storage.indexStore import IndexStore
from storage.pcaReducer import PcaDimensionReducer, IdentityDimensionReducer
from indexing.videoProcessor import VideoProcessor
from indexing.statsReporter import StatsReporter


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

        for videoPath in videoPaths:
            videoId = Path(videoPath).stem
            if videoId in processedIds:
                continue

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

        if self.config.usePcaReduction:
            dimensionReducer = PcaDimensionReducer(self.config.pcaDimension)
        else:
            dimensionReducer = IdentityDimensionReducer()

        if finalRawFeatures is not None and len(finalRawFeatures) > 0:
            dimensionReducer.fit(finalRawFeatures)
            reducedFeatures = dimensionReducer.transform(finalRawFeatures)
            self.indexStore.saveReducedFeatures(reducedFeatures)
            if self.config.usePcaReduction:
                dimensionReducer.save(self.indexStore.indexDir)

        manifest["collections"].append(collectionName)
        self.indexStore.saveManifest(manifest)

        if videoStatsList:
            self.statsReporter.reportCollection(
                collectionName, videoStatsList, dimensionReducer, self.config.usePcaReduction
            )

        return finalMetadata, videoStatsList
