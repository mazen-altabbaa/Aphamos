from test.evalMetrics import EvalMetrics
from test.evaluator import PCA_DIMENSIONS, RETRIEVAL_MODES


class EvalReporter:
    def print(self, allResults: dict) -> None:
        header = f"{'Config':<20} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'Time(s)':>8} {'Conf':>6} {'Miss':>5}"
        separator = "-" * len(header)

        for mode in RETRIEVAL_MODES:
            print(f"\n  Modality: {mode}")
            print(separator)
            print(header)
            print(separator)
            for dim in PCA_DIMENSIONS:
                label = f"pca{dim}_{mode}"
                m = allResults.get(label)
                if m is None:
                    print(f"  {label:<18}  (no data)")
                    continue
                print(
                    f"  {'PCA-'+str(dim):<18}"
                    f"  {m.recallAt1:>5.3f}"
                    f"  {m.recallAt5:>5.3f}"
                    f"  {m.recallAt10:>5.3f}"
                    f"  {m.mrr:>5.3f}"
                    f"  {m.meanQueryTimeSec:>7.3f}"
                    f"  {m.meanConfidence:>5.3f}"
                    f"  {m.notFoundCount:>4}"
                )
            print(separator)

        print(f"\n  Total queries per config: {next(iter(allResults.values())).totalQueries}")
        self._printBest(allResults)

    def _printBest(self, allResults: dict) -> None:
        bestMrr = max(allResults.items(), key=lambda x: x[1].mrr)
        bestR1 = max(allResults.items(), key=lambda x: x[1].recallAt1)
        print(f"\n  Best MRR  : {bestMrr[0]} ({bestMrr[1].mrr:.4f})")
        print(f"  Best R@1  : {bestR1[0]} ({bestR1[1].recallAt1:.4f})\n")