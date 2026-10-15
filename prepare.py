import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--anchors", nargs="*", default=[])
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.csv).open(encoding="utf-8")))
    data = {"x": np.array([[float(row[key]) for key in args.features] for row in rows]), "y": np.array([int(row["label"]) for row in rows]), "client": np.array([row["client"] for row in rows]), "split": np.array([row["split"] for row in rows])}
    if rows and "context" in rows[0]:
        data["context"] = np.array([row["context"] for row in rows])
    if args.anchors:
        data["anchors"] = np.array([args.features.index(name) for name in args.anchors])
    if not np.isfinite(data["x"]).all():
        raise ValueError("Impute missing covariates before conversion using training-only settings")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)


if __name__ == "__main__":
    main()
