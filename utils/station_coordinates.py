from pathlib import Path

import numpy as np


DEFAULT_COORDS_FILENAME = "station_coords.npy"


def load_station_coordinates(root_path, station_coords_path=None, expected_nodes=None):
    """Load ``[station, (latitude, longitude, altitude)]`` coordinates."""
    path = (
        Path(station_coords_path)
        if station_coords_path is not None
        else Path(root_path) / DEFAULT_COORDS_FILENAME
    )
    coordinates = np.load(path, allow_pickle=False)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            f"Station coordinates at {str(path)!r} must have shape [N, 3], got {coordinates.shape}."
        )
    if expected_nodes is not None and coordinates.shape[0] != expected_nodes:
        raise ValueError(
            f"Station coordinates at {str(path)!r} contain {coordinates.shape[0]} stations, expected {expected_nodes}."
        )
    if not np.isfinite(coordinates).all():
        raise ValueError(
            f"Station coordinates at {str(path)!r} contain non-finite values."
        )
    return np.asarray(coordinates, dtype=np.float32).copy()
