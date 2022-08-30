import pandas as pd
import argparse

from utils import validate_predictions
from eval_metrics import rmse, mape


def parse_args():
    parser = argparse.ArgumentParser(description="Forecasting Evaluation")
    parser.add_argument(
        "--gt-path",
        required=True,
        help="Path to source csv data",
    )
    parser.add_argument(
        "--preds-path",
        required=True,
        help="Path to source csv data",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    window_size = 12
    gt_columns = [f"oil_month_{i + 1}" for i in range(60)]
    preds_columns = [f"month_{i + 1}" for i in range(window_size)]

    gt_df = pd.read_csv(args.gt_path)
    preds_df = pd.read_csv(args.preds_path)

    validate_predictions(gt_df, preds_df)

    merged_df = pd.merge(gt_df, preds_df, on="API", how="inner")

    start_idx = int(preds_df["monthsCount"].values[0])

    gt_data = merged_df[gt_columns].iloc[:, start_idx : start_idx + window_size].values
    preds_data = merged_df[preds_columns].values

    print("RMSE:", rmse(gt_data, preds_data))
    print("MAPE:", mape(gt_data, preds_data))
