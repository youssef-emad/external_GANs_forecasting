import json
import os
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType

import numpy as np
import pandas
from scipy.optimize import curve_fit


def validate_predictions(gt_df: pandas.DataFrame, preds_df: pandas.DataFrame) -> None:
    """
    Checks if both ground-truth and predictions dataframes have identical lengths and IDs.
    """

    if len(gt_df) != len(preds_df):
        raise ValueError(
            f"Mismatching number of rows between ground-truth {len(gt_df)} and predictions {len(preds_df)}"
        )

    if set(gt_df["API"]) != set(preds_df["API"]):
        raise ValueError("Mismatching APIs between ground-truth and predictions")


def load_config_from_file(config_path: str) -> ModuleType:
    """
    Loads a configuration from a given python file path.

    Args:
        config_path (str): path to config file

    Returns:
        ModuleType: configuration module as if it's imported
    """
    spec = spec_from_file_location(
        "config",
        config_path,
    )
    config = module_from_spec(spec)
    spec.loader.exec_module(config)

    return config


def saving_criteria_satisfied(
    step_metrics: dict, saving_criteria: dict, current_best: float
) -> bool:
    """
    Checks if certain criteria is met to decided wether to save a model's checkpoint or not.

    Args:
        step_metrics (dict): results from a current step (iteration) saved as metrics' names and their computed values
        saving_criteria (dict): carries metric name and comparison mode which can be min or max.
        current_best (float): latest reached best value

    Raises:
        KeyError: if given mode (in saving_criteria) is not min or max.

    Returns:
        bool: wether the saving criteria is met or not.
    """
    metric_name = saving_criteria["metric"]

    if metric_name not in step_metrics:
        return False

    mode = saving_criteria["mode"].lower()
    step_value = step_metrics[metric_name]

    if mode == "min":
        return step_value < current_best
    elif mode == "max":
        return step_value > current_best
    else:
        raise KeyError(f"Unknow saving mode {mode}. Only min and max are allowed")


def save_config(cfg: ModuleType, output_dir: str) -> None:
    """
    Saves given configuration as JSON

    Args:
        cfg (ModuleType): configuration module to be accessed.
        output_dir (str): path to output directory to save the config as JSON file
    """

    attrs = ["experiment", "data", "generator", "discriminator", "training_params"]
    cfg_dict = {}
    for attribute in attrs:
        cfg_dict[attribute] = str(getattr(cfg, attribute, None))

    with open(os.path.join(output_dir, "experiment_parameters.json"), "w") as handle:
        json.dump(cfg_dict, handle)


def hyperbolic(t, qi, di, b):
    return qi / (np.abs((1 + b * di * t)) ** (1 / b))
