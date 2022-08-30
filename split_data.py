import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Data Split")
    parser.add_argument(
        "--source-csv",
        required=True,
        help="Path to source csv data",
    )
    parser.add_argument(
        "--test-csv",
        required=True,
        help="Path to source csv data",
    )
    parser.add_argument(
        "--output-dir",
        default="split_files",
        help="Path to output directory",
    )

    parser.add_argument(
        "--select-features",
        default=None,
        help="Select Features. oil or gas",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_df = pd.read_csv(args.source_csv)
    test_apis_df = pd.read_csv(args.test_csv)

    train_df = data_df[~data_df["API"].isin(test_apis_df["API"])]
    test_df = pd.merge(data_df, test_apis_df[["API"]], on="API", how="inner")

    if args.select_features:
        feature_columns = [
            "API",
            f"{args.select_features}Di",
            f"{args.select_features}Qi",
            "b",
        ] + [f"{args.select_features}_month_{i + 1}" for i in range(60)]
        train_df = train_df[feature_columns]
        test_df = test_df[feature_columns]

    train_df.to_csv(os.path.join(args.output_dir, "train_data.csv"))
    test_df.to_csv(os.path.join(args.output_dir, "test_data.csv"))
