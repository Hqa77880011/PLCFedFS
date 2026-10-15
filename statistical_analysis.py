import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--csv", required=True); parser.add_argument("--reference", default="full"); parser.add_argument("--output", required=True); parser.add_argument("--seed", type=int, default=7); args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.csv).open(encoding="utf-8")))
    methods = sorted({r["method"] for r in rows}); blocks = sorted({r["block"] for r in rows})
    values = np.array([[np.mean([float(r["value"]) for r in rows if r["block"] == block and r["method"] == method]) for method in methods] for block in blocks])
    if not np.isfinite(values).all() or len(methods) < 3 or len(blocks) < 3:
        raise ValueError("Every independent dataset-shift block needs every method; at least three methods/blocks")
    reference = methods.index(args.reference); omnibus = friedmanchisquare(*values.T)
    rng = np.random.default_rng(args.seed); comparisons = []
    for j, method in enumerate(methods):
        if j == reference:
            continue
        difference = values[:, reference] - values[:, j]
        nonzero = difference[difference != 0]
        if len(nonzero) > 20:
            raise ValueError("Exact signed-rank enumeration supports at most 20 nonzero independent blocks")
        ranks = rankdata(np.abs(nonzero))
        observed = abs(float(np.sum(np.sign(nonzero) * ranks)))
        signs = 2 * ((np.arange(2 ** len(nonzero))[:, None] >> np.arange(len(nonzero))) & 1) - 1
        distribution = np.abs(signs @ ranks)
        p = float(np.mean(distribution >= observed - 1e-10))
        samples = rng.choice(difference, (10000, len(difference)), replace=True).mean(1)
        comparisons.append({"method": method, "raw_p": p, "median_difference": float(np.median(difference)), "wins": int(np.sum(difference > 0)), "mean_difference_95_ci": np.quantile(samples, [.025, .975]).tolist()})
    ordered = sorted(comparisons, key=lambda r: r["raw_p"]); running = 0.
    for rank, row in enumerate(ordered):
        running = max(running, min(1., row["raw_p"] * (len(ordered) - rank))); row["holm_p"] = running
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"friedman_statistic": float(omnibus.statistic), "friedman_p": float(omnibus.pvalue), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
