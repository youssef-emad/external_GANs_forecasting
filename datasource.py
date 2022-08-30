from random import choice
from typing import Callable, List

import numpy
import numpy as np
import pandas
from torch.utils.data import Dataset


class DataSource(Dataset):
    """
    Datasource wrapper for time-series dataset
    """

    def __init__(
        self,
        df: pandas.DataFrame,
        min_window: int,
        sampler: Callable = None,
        augmentations: List[Callable] = None,
    ):
        """
        Initializes a DataSource instance

        Args:

            df (pandas.DataFrame): dataframe carrying timeseris data expected to have an API column
                followed by the time-seris points (features)
            min_window (int): length of minimum number of features (points) that must exist in an instance
                of the dataset.
            sampler (Callable): sampling function that's called at getting an item. If not set, the entire entity
                is returned.
            augmentations (List(Callable)): list of augmentations functions to be applied at getting an item.
        """
        self.sampler = sampler
        self.augmentations = augmentations
        self.mean = self.std = None
        self.data, self.data_len, self.ids = self.prepare_data(df, min_window)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        if self.sampler:
            sample = self.sampler(
                self.data[index], self.data_len[index], self.ids[index]
            )
        else:
            sample = self.data[index]

        # randomly select on the augmentations
        if self.augmentations:
            aug_func = choice(self.augmentations)
            sample = aug_func(sample)
        return sample

    def prepare_data(
        self, df: pandas.DataFrame, min_window: int
    ) -> tuple[numpy.array, numpy.array, List[str]]:
        """
        Parses the time-series data from a given dataframe, computes its mean and std, then returns the data as numpy array.
        It starts by removing any rows with number features less than the given minimum window.
        It also returns additional information as the length of each entity and it's ID (API)

        Args:
            df (pandas.DataFrame): dataframe carrying timeseris data expected to have an API column
                followed by the time-seris points (features)
            min_window (int): length of minimum number of features (points) that must exist in an instance
                of the dataset.

        Returns:
           numpy.array, numpy.array, List[str]: time-seris entites, time-series lengths, times-series ids.
        """

        # compute the maximum number of missing points
        total_window_len = df.shape[1]
        max_missing = total_window_len - min_window

        # select rows that satisfies the condition
        df["num_zeros"] = (df == 0).astype(int).sum(axis=1)
        df["num_nans"] = df.isnull().sum(axis=1)
        df = df[(df["num_nans"] <= max_missing) & (df["num_zeros"] <= max_missing)]

        # save ids
        ids = df["API"].values
        df = df.drop("API", axis=1)

        # fill missing values with zeros
        df = df.fillna(0.0)

        # save the data as numpy arrays
        data = df.iloc[:, :-2].values.astype(np.float32)
        data_len = total_window_len - df.iloc[:, -1].values

        # computes and saves the dataset mean and standard deviatoin
        self.mean = np.mean(data)
        self.std = np.std(data)

        return data, data_len, ids
