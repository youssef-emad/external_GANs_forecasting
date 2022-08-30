from generators import G_CNN_RNN
from utils import load_config_from_file, save_config

import pandas as pd
from datasource import DataSource
from torch.utils.data import DataLoader
import argparse
import torch
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Background Mining")
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to python configuration file",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Path to output directory",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    # check available GPUs
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")

    # load configuration
    cfg = load_config_from_file(args.config_path)

    # create output directories
    output_dir = os.path.join(args.output_dir, cfg.experiment["name"])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    # save configuration
    save_config(cfg, output_dir)

    # load training data
    train_df = pd.read_csv(cfg.data["training"]["csv_path"])
    train_df = train_df[["API"] + cfg.experiment["feature_columns"]]
    training_dataset = DataSource(
        df=train_df,
        min_window=cfg.experiment["min_window_size"],
        sampler=cfg.data["training"]["sampler"],
    )
    training_dataloader = DataLoader(
        training_dataset, **cfg.data["training"]["dataloader_params"]
    )

    # load validation data
    validation_df = pd.read_csv(cfg.data["validation"]["csv_path"])
    validation_df = validation_df[["API"] + cfg.experiment["feature_columns"]]
    validation_dataset = DataSource(
        df=validation_df,
        min_window=cfg.experiment["min_window_size"],
        sampler=cfg.data["validation"]["sampler"],
    )
    validation_dataloader = DataLoader(
        validation_dataset, **cfg.data["validation"]["dataloader_params"]
    )

    # define generator
    generator = G_CNN_RNN(
        out_window_size=cfg.experiment["output_window_size"],
        condition_size=cfg.experiment["input_window_size"],
        noise_size=0,
        latent_size=cfg.generator["latent_size"],
        rnn_cell_type=cfg.generator["rnn_cell_type"],
        n_filters=cfg.generator["n_filters"],
        kernel_size=cfg.generator["kernel_size"],
        pool_size=cfg.generator["pool_size"],
        mean=training_dataset.mean,
        std=training_dataset.std,
    ).to(DEVICE)

    # start training
    generator.train(
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        training_params=cfg.training_params,
        output_dir=output_dir,
        device=DEVICE,
    )
