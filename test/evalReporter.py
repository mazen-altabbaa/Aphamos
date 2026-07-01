from test.evalMetrics import EvalMetrics


class EvalReporter:
    def print(self, metrics: EvalMetrics) -> None:
        print("\n" + "=" * 20)
        print("  Evaluation Results")
        print("=" * 20)
        print(f"  Total queries   : {metrics.totalQueries}")
        print(f"  Not found       : {metrics.notFoundCount}")
        print(f"  R@1             : {metrics.recallAt1:.4f}")
        print(f"  R@5             : {metrics.recallAt5:.4f}")
        print(f"  R@10            : {metrics.recallAt10:.4f}")
        print(f"  MRR             : {metrics.mrr:.4f}")
        print(f"  Mean query time : {metrics.meanQueryTimeSec:.3f}s")
        print(f"  Mean confidence : {metrics.meanConfidence:.4f}")
        print("=" * 20 + "\n")