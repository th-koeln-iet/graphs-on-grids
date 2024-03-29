import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculates the mean squared error (MSE) based on the average MSE on all nodes

    :param y_true: Correct target values
    :param y_pred: Predicted target values
    :return: MSE for each node feature averaged over all nodes and instances
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Invalid dimensions between y_true and y_pred. Got shapes {y_true.shape} and {y_pred.shape}"
        )
    if len(y_true.shape) == 2:
        output = np.average((y_true - y_pred) ** 2, axis=0)
    elif len(y_true.shape) == 3:
        output = [
            np.average((y - y_hat) ** 2, axis=0) for y, y_hat in zip(y_true, y_pred)
        ]
        output = np.average(output, axis=0)
    else:
        raise ValueError(
            f"Invalid dimension {len(y_true.shape)} provided. Either 2d or 3d array expected."
        )
    return output


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculates the root mean squared error (RMSE) based on the average RMSE on all nodes

    :param y_true: Correct target values
    :param y_pred: Predicted target values
    :return: RMSE for each node feature averaged over all nodes and instances
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Invalid dimensions between y_true and y_pred. Got shapes {y_true.shape} and {y_pred.shape}"
        )
    if len(y_true.shape) == 2:
        output = np.average((y_true - y_pred) ** 2, axis=0)
    elif len(y_true.shape) == 3:
        output = [
            np.average((y - y_hat) ** 2, axis=0) for y, y_hat in zip(y_true, y_pred)
        ]
        output = np.average(np.sqrt(output), axis=0)
    else:
        raise ValueError(
            f"Invalid dimension {len(y_true.shape)} provided. Either 2d or 3d array expected."
        )
    return output


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculates the mean absolute error (MAE) based on the average MAE on all nodes

    :param y_true: Correct target values
    :param y_pred: Predicted target values
    :return: MAE for each node feature averaged over all nodes and instances
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Invalid dimensions between y_true and y_pred. Got shapes {y_true.shape} and {y_pred.shape}"
        )

    if len(y_true.shape) == 2:
        output = np.average(np.abs(y_true - y_pred), axis=0)
    elif len(y_true.shape) == 3:
        output = [
            np.average(np.abs(y - y_hat), axis=0) for y, y_hat in zip(y_true, y_pred)
        ]
        output = np.average(output, axis=0)
    else:
        raise ValueError(
            f"Invalid dimension {len(y_true.shape)} provided. Either 2d or 3d array expected."
        )
    return output
