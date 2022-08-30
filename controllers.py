from random import randint
from typing import List

import numpy as np


def sampler(
    data: np.array,
    data_len: np.array,
    id: List[str],
    input_window_size: int,
    output_window_size: int,
    mode: str = "random",
) -> tuple[np.array, np.array, dict]:
    """
    Returns a sample from a given time-series entity

    Args:
        data (numpy.array): 1D numpy array (or list) represent a time-series data to be sampled from
        data_len (int): Number of available points (not-nans). It can also be considered as the last available point
        id (str): ID of the given time-seris
        input_window_size (int): size of input window
        output_window_size (int): size of output (forecasted) window
        mode (str): sampling mode wether deterministic (select from the beginning of the time-series)
            or random (select from a valid random starting point)

    Returns:
        tuple (numpy.array, numpy.array, dict): input time-seris data, output time-series data, metadata dictionary
    """

    if mode not in ["random", "deterministic"]:
        raise ValueError(
            f"Invalid sampling mode ({mode}). Only random and deterministic are available"
        )

    total_window_size = input_window_size + output_window_size

    if data_len <= total_window_size or mode == "deterministic":
        start = 0

    elif mode == "random":
        start = randint(0, data_len - total_window_size - 1)

    sample = data[start : start + total_window_size]

    meta = {
        "start": start,
        "end": start + total_window_size,
        "input_window_size": input_window_size,
        "output_window_size": output_window_size,
        "id": id,
    }

    return sample[:input_window_size], sample[input_window_size:], meta
