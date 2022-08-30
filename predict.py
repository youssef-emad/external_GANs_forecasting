import argparse
import pandas as pd
from utils import load_config_from_file
from datasource import DataSource
from torch.utils.data import DataLoader
from functools import partial
import torch
from controllers import sampler
from eval_metrics import evaluate
from tqdm import tqdm
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description="GANs Forecasting")
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to python configuration file",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to source data excel",
    )
    parser.add_argument(
        "--pth-path",
        default=None,
        help="Path to model pth",
    )
    parser.add_argument(
        "--pred-steps",
        default=10,
        help="Number of steps for generator prediction",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Path to output directory",
    )
    parser.add_argument(
        "--sampler-mode",
        default="deterministic",
        help="Sampling mode. random or deterministic",
    )
    parser.add_argument(
        "--dummy-mode",
        action="store_true",
        help="Dummy (Mean) prediction mode",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    # set seed for reproducibility
    torch.manual_seed(10)

    # check available GPUs
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")

    # load configuration
    cfg = load_config_from_file(args.config_path)

    # load model
    if args.dummy_mode and args.pth_path is not None:
        raise argparse.ArgumentError("Passed both pth-path and dummy-mode")

    # define test sampler
    test_sampler = partial(
        sampler,
        input_window_size=cfg.input_window_size,
        output_window_size=cfg.output_window_size,
        mode=args.sampler_mode,
    )

    # load test data
    test_df = pd.read_csv(args.data_path)
    test_df = test_df[["API"] + cfg.experiment["feature_columns"]]
    test_dataset = DataSource(
        df=test_df, min_window=cfg.min_window_size, sampler=test_sampler
    )
    test_dataloader = DataLoader(test_dataset, batch_size=60, shuffle=False)

    # load model (if given)
    if args.pth_path:
        model = torch.load(args.pth_path)
        saving_name = os.path.basename(args.pth_path).replace(".pth", ".csv")
    else:
        saving_name = "dummy.csv"

    # create predictions dataframe and necessary lists
    prediction_df = pd.DataFrame(
        columns=["API", "start", "end"]
        + [f"month_{i + 1}" for i in range(cfg.output_window_size)]
    )

    preds = []
    apis = []
    gt = []
    starts = []
    ends = []

    # inference loop
    for x_test, y_test, meta in tqdm(test_dataloader):

        start = (meta["start"] + meta["input_window_size"] + 1).detach().cpu().numpy()
        end = meta["end"].detach().cpu().numpy()
        api = meta["id"].detach().cpu().numpy()

        starts.extend(start)
        ends.extend(end)
        apis.extend(api)

        if args.dummy_mode:
            y_pred = np.mean(x_test.detach().cpu().numpy(), axis=1)
            # repeat mean across the whole output window
            y_pred = np.tile(
                np.array([y_pred]).transpose(), (1, cfg.output_window_size)
            )

        elif args.pth_path:
            y_pred = []
            for _ in range(args.pred_steps):
                noise = torch.normal(
                    0,
                    1,
                    (x_test.shape[0], cfg.generator["noise_window_size"]),
                    device=DEVICE,
                    dtype=torch.float32,
                )
                y_pred.append(model(x_test, noise).detach().cpu().numpy())

            y_pred = np.mean(np.array(y_pred), axis=0)
        preds.extend(y_pred)
        gt.extend(y_test.detach().cpu().numpy())

    # save predictions in csv
    prediction_df["API"] = apis
    prediction_df["start"] = starts
    prediction_df["end"] = ends
    prediction_df[[f"month_{i + 1}" for i in range(cfg.output_window_size)]] = preds
    prediction_df.to_csv(os.path.join(args.output_dir, saving_name))

    # run evaluation
    result = evaluate(gt, preds)

    print(result)
