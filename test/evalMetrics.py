from dataclasses import dataclass


@dataclass
class EvalMetrics:
    recallAt1: float
    recallAt5: float
    recallAt10: float
    mrr: float
    meanQueryTimeSec: float
    meanConfidence: float
    totalQueries: int
    notFoundCount: int


class MetricsCalculator:
    def compute(self, queryResults: list) -> EvalMetrics:
        reciprocalRanks = []
        hits1 = hits5 = hits10 = 0
        totalTimeSec = 0.0
        totalConfidence = 0.0
        notFoundCount = 0

        for result in queryResults:
            rank = result["rank"]
            totalTimeSec += result["queryTimeSec"]
            totalConfidence += result["confidence"]

            if rank is None:
                notFoundCount += 1
                reciprocalRanks.append(0.0)
                continue

            reciprocalRanks.append(1.0 / rank)
            if rank <= 1:
                hits1 += 1
            if rank <= 5:
                hits5 += 1
            if rank <= 10:
                hits10 += 1

        n = len(queryResults)
        return EvalMetrics(
            recallAt1=hits1 / n,
            recallAt5=hits5 / n,
            recallAt10=hits10 / n,
            mrr=sum(reciprocalRanks) / n,
            meanQueryTimeSec=totalTimeSec / n,
            meanConfidence=totalConfidence / n,
            totalQueries=n,
            notFoundCount=notFoundCount,
        )