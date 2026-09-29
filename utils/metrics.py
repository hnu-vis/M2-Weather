import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {pred.shape} vs {true.shape}."
        )
    if pred.ndim == 0:
        return float((np.float64(true) - np.float64(pred)) ** 2)

    squared_error_sum = 0.0
    element_count = 0
    for start in range(0, pred.shape[0], 16):
        difference = np.subtract(
            pred[start:start + 16], true[start:start + 16], dtype=np.float64
        )
        squared_error_sum += np.sum(difference * difference, dtype=np.float64)
        element_count += difference.size
    return squared_error_sum / element_count


def mse_by_variable(pred, true, num_stations, num_variables):
    """Compute one MSE per variable from station-major flattened outputs."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {pred.shape} vs {true.shape}."
        )

    expected_channels = num_stations * num_variables
    if pred.shape[-1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} flattened output channels "
            f"({num_stations} stations x {num_variables} variables), "
            f"but received {pred.shape[-1]}."
        )

    squared_error_sum = np.zeros(num_variables, dtype=np.float64)
    element_count = 0
    for start in range(0, pred.shape[0], 16):
        difference = np.subtract(
            pred[start:start + 16], true[start:start + 16], dtype=np.float64
        ).reshape(
            *pred[start:start + 16].shape[:-1], num_stations, num_variables
        )
        reduction_axes = tuple(range(difference.ndim - 1))
        squared_error_sum += np.sum(
            difference * difference, axis=reduction_axes, dtype=np.float64
        )
        element_count += difference.size // num_variables
    return squared_error_sum / element_count


def mae_by_variable(pred, true, num_stations, num_variables):
    """Compute one MAE per variable from station-major flattened outputs."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {pred.shape} vs {true.shape}."
        )

    expected_channels = num_stations * num_variables
    if pred.shape[-1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} flattened output channels "
            f"({num_stations} stations x {num_variables} variables), "
            f"but received {pred.shape[-1]}."
        )

    absolute_error_sum = np.zeros(num_variables, dtype=np.float64)
    element_count = 0
    for start in range(0, pred.shape[0], 16):
        difference = np.subtract(
            pred[start:start + 16], true[start:start + 16], dtype=np.float64
        ).reshape(
            *pred[start:start + 16].shape[:-1], num_stations, num_variables
        )
        reduction_axes = tuple(range(difference.ndim - 1))
        absolute_error_sum += np.sum(
            np.abs(difference), axis=reduction_axes, dtype=np.float64
        )
        element_count += difference.size // num_variables
    return absolute_error_sum / element_count


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = np.sqrt(mse)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
