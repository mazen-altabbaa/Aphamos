import time
import numpy as np
from storage.indexStore import IndexStore
from storage.pcaReducer import PcaDimensionReducer, IdentityDimensionReducer


class QuerySearch:
    def __init__(self, config, visionEncoder):
        self.config = config
        self.visionEncoder = visionEncoder
        self.indexStore = IndexStore(config.indexDir)

    def queryWithText(self, textQuery: str, topK: int = 5):
        return self._query(lambda: self.visionEncoder.encodeText(textQuery), topK)

    def queryWithImage(self, frame, topK: int = 5):
        return self._query(lambda: self.visionEncoder.encodeImage(frame), topK)

    def _query(self, embedFn, topK: int):
        timings = {}

        startTime = time.perf_counter()
        rawQuery = embedFn()
        timings["embedQuerySec"] = time.perf_counter() - startTime

        startTime = time.perf_counter()
        reducedFeatures = self.indexStore.loadReducedFeatures()
        metadata = self.indexStore.loadMetadata()
        timings["loadIndexSec"] = time.perf_counter() - startTime

        startTime = time.perf_counter()
        if self.config.usePcaReduction:
            reducer = PcaDimensionReducer.load(self.indexStore.indexDir, self.config.pcaDimension)
        else:
            reducer = IdentityDimensionReducer()
        transformedQuery = reducer.transform(rawQuery.reshape(1, -1)).flatten()
        timings["transformQuerySec"] = time.perf_counter() - startTime

        startTime = time.perf_counter()
        normalizedQuery = transformedQuery / (np.linalg.norm(transformedQuery) + 1e-10)
        normalizedIndex = reducedFeatures / (np.linalg.norm(reducedFeatures, axis=1, keepdims=True) + 1e-10)
        similarities = normalizedIndex @ normalizedQuery
        weights = np.array([item.get("appliedWeight", 1.0) for item in metadata])
        weightedSimilarities = similarities * weights
        timings["similaritySec"] = time.perf_counter() - startTime

        startTime = time.perf_counter()
        rankedIndices = np.argsort(weightedSimilarities)[::-1]
        results, seenVideoIds = [], set()
        for index in rankedIndices:
            videoId = metadata[index]["videoId"]
            if videoId in seenVideoIds:
                continue
            seenVideoIds.add(videoId)
            results.append({
                "videoId": videoId,
                "score": float(weightedSimilarities[index]),
                "matchedItem": metadata[index],
            })
            if len(results) >= topK:
                break
        timings["rankAndDedupeSec"] = time.perf_counter() - startTime

        timings["totalSec"] = sum(timings.values())
        return results, timings
