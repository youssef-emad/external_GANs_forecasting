import numpy as np
from sklearn.metrics import accuracy_score


def rmse(y_true: np.array, y_pred: np.array) -> float:
    """
    Root Mean Sqaured Error

    Args:
        y_true (numpy.array): ground-truth continous values
        y_pred (numpy.array): predicted continous values

    Returns:
        float: calculated RMSE
    """
    if y_pred.ndim == 3:
        y_pred = np.mean(y_pred, axis=1)
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def mae(y_true: np.array, y_pred: np.array) -> float:
    """
    Mean Absolute Error

    Args:
        y_true (numpy.array): ground-truth continous values
        y_pred (numpy.array): predicted continous values

    Returns:
        float: calculated MAE
    """
    if y_pred.ndim == 3:
        y_pred = np.mean(y_pred, axis=1)

    mask = np.where(y_true != 0)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return np.fabs(y_true - y_pred).mean()


def mape(y_true: np.array, y_pred: np.array) -> float:
    """
    Mean Absolute Percentage Error

    Args:
        y_true (numpy.array): ground-truth continous values
        y_pred (numpy.array): predicted continous values

    Returns:
        float: calculated MAPE
    """
    if y_pred.ndim == 3:
        y_pred = np.mean(y_pred, axis=1)

    mask = np.where(y_true != 0)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return (np.fabs(y_true - y_pred) / y_true).mean()


def picp(y_true: np.array, y_pred: np.array) -> float:
    """
    Prediction Interval Coverage Probability

    Args:
        y_true (numpy.array): ground-truth continous values
        y_pred (numpy.array): predicted continous values

    Returns:
        float: calculated PICP
    """
    y_true_upper = y_true <= np.max(y_pred, axis=1)
    y_truelower = y_true >= np.min(y_pred, axis=1)
    return np.mean(y_true_upper & y_truelower)


def pinaw(y_true: np.array, y_pred: np.array, range: int = 1) -> float:
    """
    Prediction Interval Normalized Average Width

    Args:
        y_true (numpy.array): ground-truth continous values
        y_pred (numpy.array): predicted continous values
        range (int): prediction range

    Returns:
        float: calculated PINAW
    """
    pred_interval = np.max(y_pred, axis=1) - np.min(y_pred, axis=1)
    return np.mean(pred_interval) / range


def evaluate(y_true: np.array, y_pred: np.array) -> dict:
    """
    Calculate multiple evaluation metrics for given ground-truth and predictions
        The current metrics are RMSE, MAPE and MAE

    Args:
        y_true (numpy.array): ground-truth continous values
        y_pred (numpy.array): predicted continous values

    Returns:
        dict: result dictionary mapping between metric names (rmse, mape, mae) as keys
            and their computed values.
    """

    metrics = ["rmse", "mape", "mae"]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    result = {}
    for metric in metrics:
        metric_func = metrics_mapper[metric]
        result[metric] = metric_func(y_true, y_pred)
    return result


# maps between metric names and their functions
metrics_mapper = {
    "rmse": rmse,
    "mape": mape,
    "mae": mae,
    "picp": picp,
    "pinaw": pinaw,
    "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred > 0.5),
}
