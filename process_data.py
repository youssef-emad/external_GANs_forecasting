import argparse
import os

import numpy as np
import pandas as pd

from utils import hyperbolic


def parse_args():
    parser = argparse.ArgumentParser(description="Data Split")
    parser.add_argument(
        "--source-csv",
        required=True,
        help="Path to source csv data",
    )

    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to pre-processed csv data",
    )
    parser.add_argument(
        "--mode",
        required=True,
        help="Processing Mode. pre or post",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.mode.lower() not in ["pre", "post"]:
        raise argparse.ArgumentError("Mode must be pre or post")

    df = pd.read_csv(args.source_csv)

    MAX_TS = 60

    feature_cols = [f"oil_month_{i + 1}" for i in range(MAX_TS)]
    dca_curve = (
        df[["oilQi", "oilDi", "b"]]
        .apply(lambda x: hyperbolic(np.arange(MAX_TS), *x), axis=1)
        .values
    )

    dca_curve = np.array([row for row in dca_curve])
    if args.mode.lower() == "pre":
        df[feature_cols] = df[feature_cols] - dca_curve
    elif args.mode.lower() == "post":
        df[feature_cols] = df[feature_cols] + dca_curve
    df.to_csv(args.output_csv)
