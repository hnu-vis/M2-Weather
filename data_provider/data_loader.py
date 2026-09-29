import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
STATE_NAMES = ("u", "v", "T", "RH")
M3_FRANCE_STATE_NAMES = ("T", "WS", "RH", "P")


def weather2k_raw_to_state_np(raw):
    """Convert raw channels to the four directly observed forecast states."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[-1] != 13:
        raise ValueError(f"Expected 13 raw Weather2K channels, got {raw.shape[-1]}.")

    temperature_c = raw[..., 4]
    relative_humidity = np.clip(raw[..., 7], 0.0, 100.0)
    wind_direction = np.deg2rad(raw[..., 9])
    wind_speed = raw[..., 10]
    u = -wind_speed * np.sin(wind_direction)
    v = -wind_speed * np.cos(wind_direction)
    return np.stack((u, v, temperature_c, relative_humidity), axis=-1).astype(
        np.float32
    )


def m3_france_raw_to_state_np(raw):
    """Convert M3 France channels to ``[T, WS, RH, P]``.

    Relative humidity is derived from air/dew-point temperature with the
    Magnus formula. Pressure is already stored in hPa by the dataset builder.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[-1] != 8:
        raise ValueError(f"Expected 8 raw M3 France channels, got {raw.shape[-1]}.")
    temperature = raw[..., 3]
    dew_point = raw[..., 4]
    wind_speed = raw[..., 5]
    pressure = raw[..., 6]
    a, b = np.float32(17.625), np.float32(243.04)
    exponent = (
        a * dew_point / (b + dew_point)
        - a * temperature / (b + temperature)
    )
    relative_humidity = np.clip(
        100.0 * np.exp(np.clip(exponent, -50.0, 50.0)), 0.0, 100.0
    )
    return np.stack(
        (temperature, wind_speed, relative_humidity, pressure), axis=-1
    ).astype(np.float32)


class _RepeatedFeatureScaler:
    """Sklearn-compatible inverse scaler for station-repeated variables."""

    def __init__(self, mean, scale, num_stations):
        self.state_mean_ = np.asarray(mean, dtype=np.float32)
        self.state_scale_ = np.asarray(scale, dtype=np.float32)
        self.mean_ = np.tile(self.state_mean_, int(num_stations))
        self.scale_ = np.tile(self.state_scale_, int(num_stations))
        self.var_ = self.scale_ ** 2
        self.n_features_in_ = self.mean_.size

    def inverse_transform(self, data):
        data = np.asarray(data)
        if data.shape[-1] == self.n_features_in_:
            return data * self.scale_ + self.mean_
        if data.shape[-1] == self.state_mean_.size:
            return data * self.state_scale_ + self.state_mean_
        raise ValueError(
            f"Cannot inverse-transform {data.shape[-1]} features; expected "
            f"{self.n_features_in_} flattened or {self.state_mean_.size} state features."
        )


class Dataset_French(Dataset):
    """Hourly station data stored as ``[station, 13, time]``.

    The dynamic channels are converted to ``[u, v, T, RH]``. Windows
    retain the canonical ``[time, station, variable]`` layout; each model
    wrapper performs its own batch-axis merge.
    """

    _stats_cache = {}

    def __init__(self, args, root_path, flag="train", size=None,
                 data_path="meteonet_nw_1h_2016_2018_nopsl_weather2k.npy",
                 scale=True, freq="h"):
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported French split: {flag!r}.")
        self.args = args
        self.flag = flag
        self.seq_len, self.label_len, self.pred_len = size
        self.scale = bool(scale)
        self.freq = freq
        self.data_path = (
            data_path if os.path.isabs(data_path) else os.path.join(root_path, data_path)
        )
        self.raw = np.load(self.data_path, mmap_mode="r", allow_pickle=False)
        if self.raw.ndim != 3 or self.raw.shape[1] != 13:
            raise ValueError(
                "Dataset_French expects [station, 13, time], "
                f"but {self.data_path!r} has shape {self.raw.shape}."
            )

        self.num_stations = int(self.raw.shape[0])
        requested_variables = getattr(args, "state_variables", None)
        if requested_variables is None:
            requested_variables = STATE_NAMES
        if len(set(requested_variables)) != len(requested_variables):
            raise ValueError(
                f"French state_variables contains duplicates: {requested_variables}."
            )
        self.variable_names = tuple(requested_variables)
        self.variable_indices = np.asarray(
            [STATE_NAMES.index(name) for name in self.variable_names], dtype=np.int64
        )
        self.num_features = len(self.variable_names)
        self.num_time_steps = int(self.raw.shape[2])
        self.station_coords = np.asarray(self.raw[:, :3, 0], dtype=np.float32).copy()

        train_end = int(self.num_time_steps * 0.6)
        val_end = int(self.num_time_steps * 0.7)
        bounds = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, self.num_time_steps),
        }
        self.split_start, self.split_end = bounds[flag]
        if self.split_end - self.split_start < self.seq_len + self.pred_len:
            raise ValueError(
                f"{flag} split is too short for seq_len={self.seq_len} and "
                f"pred_len={self.pred_len}."
            )

        if self.scale:
            mean, std = self._state_stats(train_end)
            mean = mean[self.variable_indices]
            std = std[self.variable_indices]
        else:
            mean = np.zeros(self.num_features, dtype=np.float32)
            std = np.ones(self.num_features, dtype=np.float32)
        self.state_mean = mean
        self.state_std = std
        self.mean_out = mean.tolist()
        self.std_out = std.tolist()
        self.scaler = _RepeatedFeatureScaler(mean, std, self.num_stations)

    def _state_stats(self, train_end, chunk_size=256):
        cache_key = (os.path.realpath(self.data_path), train_end)
        cached = self._stats_cache.get(cache_key)
        if cached is not None:
            return cached
        # Cache canonical four-state statistics independently of the selected
        # training subset, then slice them in ``__init__``.
        total = np.zeros(len(STATE_NAMES), dtype=np.float64)
        total_sq = np.zeros(len(STATE_NAMES), dtype=np.float64)
        count = np.zeros(len(STATE_NAMES), dtype=np.float64)
        for start in range(0, train_end, chunk_size):
            stop = min(start + chunk_size, train_end)
            block = np.asarray(self.raw[:, :, start:stop], dtype=np.float32)
            state = weather2k_raw_to_state_np(np.moveaxis(block, -1, 0))
            finite = np.isfinite(state)
            values = np.where(finite, state, 0.0)
            total += values.sum(axis=(0, 1), dtype=np.float64)
            total_sq += np.square(values).sum(axis=(0, 1), dtype=np.float64)
            count += finite.sum(axis=(0, 1), dtype=np.float64)
        mean = total / np.maximum(count, 1.0)
        variance = total_sq / np.maximum(count, 1.0) - np.square(mean)
        std = np.sqrt(np.maximum(variance, 1e-12))
        std[std < 1e-6] = 1.0
        result = mean.astype(np.float32), std.astype(np.float32)
        self._stats_cache[cache_key] = result
        return result

    def _state_window(self, start, length):
        block = np.asarray(self.raw[:, :, start:start + length], dtype=np.float32)
        state = weather2k_raw_to_state_np(np.moveaxis(block, -1, 0))
        state = state[..., self.variable_indices]
        if self.scale:
            state = (
                (state - self.state_mean.reshape(1, 1, -1))
                / self.state_std.reshape(1, 1, -1)
            )
        return state

    @staticmethod
    def _time_marks(start, length):
        dates = pd.Timestamp("2016-01-01") + pd.to_timedelta(
            np.arange(start, start + length), unit="h"
        )
        return np.stack(
            (
                (dates.month - 1) / 12.0,
                (dates.day - 1) / 31.0,
                dates.hour / 24.0,
                (dates.dayofyear - 1) / 366.0,
            ),
            axis=-1,
        ).astype(np.float32)

    def __getitem__(self, index):
        s_begin = self.split_start + int(index)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_length = self.label_len + self.pred_len
        seq_x = self._state_window(s_begin, self.seq_len)
        seq_y = self._state_window(r_begin, r_length)
        seq_x_mark = self._time_marks(s_begin, self.seq_len)
        seq_y_mark = self._time_marks(r_begin, r_length)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, s_end

    def __len__(self):
        return self.split_end - self.split_start - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M3France(Dataset):
    """M3 France hourly data in canonical ``[time, station, variable]`` form.

    The source array is ``[station, 8, time]`` and the forecast variables are
    ``[T, WS, RH, P]``. The chronological 60/10/30 split and window semantics
    intentionally match :class:`Dataset_French`.
    """

    _stats_cache = {}
    _calendar_cache = {}

    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        data_path="m3_france_france_q0q1_weather5_1h_2017_2021.npy",
        scale=True,
        freq="h",
        split_role=None,
    ):
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported M3 France split: {flag!r}.")
        self.args = args
        self.flag = flag
        self.seq_len, self.label_len, self.pred_len = size
        self.scale = bool(scale)
        self.freq = freq
        self.root_path = os.path.realpath(root_path)
        self.data_path = (
            data_path if os.path.isabs(data_path)
            else os.path.join(root_path, data_path)
        )
        self.raw = np.load(self.data_path, mmap_mode="r", allow_pickle=False)
        if self.raw.ndim != 3 or self.raw.shape[1] != 8:
            raise ValueError(
                "Dataset_M3France expects [station, 8, time], "
                f"but {self.data_path!r} has shape {self.raw.shape}."
            )

        self.num_stations = int(self.raw.shape[0])
        self.num_time_steps = int(self.raw.shape[2])
        requested_variables = getattr(args, "state_variables", None)
        if requested_variables is None:
            requested_variables = M3_FRANCE_STATE_NAMES
        if len(set(requested_variables)) != len(requested_variables):
            raise ValueError(
                f"M3 France state_variables contains duplicates: {requested_variables}."
            )
        unknown = set(requested_variables).difference(M3_FRANCE_STATE_NAMES)
        if unknown:
            raise ValueError(
                f"Unknown M3 France variables {sorted(unknown)}; choose from "
                f"{M3_FRANCE_STATE_NAMES}."
            )
        self.variable_names = tuple(requested_variables)
        self.variable_indices = np.asarray(
            [M3_FRANCE_STATE_NAMES.index(name) for name in self.variable_names],
            dtype=np.int64,
        )
        self.num_features = len(self.variable_names)

        coordinate_path = os.path.join(root_path, "station_coords.npy")
        self.station_coords = np.asarray(
            np.load(coordinate_path, allow_pickle=False), dtype=np.float32
        )
        if self.station_coords.shape != (self.num_stations, 3):
            raise ValueError(
                f"Expected station_coords.npy shape {(self.num_stations, 3)}, "
                f"got {self.station_coords.shape}."
            )

        time_path = os.path.join(root_path, "time.npy")
        self.times = np.asarray(np.load(time_path, allow_pickle=False)).astype(
            "datetime64[h]"
        )
        if self.times.shape != (self.num_time_steps,):
            raise ValueError(
                f"Expected time.npy shape {(self.num_time_steps,)}, got {self.times.shape}."
            )
        if self.num_time_steps > 1 and not np.all(
            np.diff(self.times).astype("timedelta64[h]").astype(np.int64) == 1
        ):
            raise ValueError("M3 France time.npy must be a continuous hourly axis.")
        reference = np.datetime64("2016-01-01T00", "h")
        self.forecast_hour_offset = int(
            (self.times[0] - reference).astype("timedelta64[h]").astype(np.int64)
        )
        self.calendar_marks = self._calendar_marks(time_path)

        train_end = int(self.num_time_steps * 0.6)
        val_end = int(self.num_time_steps * 0.7)
        profile = getattr(args, "m3_split_profile", "default")
        if profile == "adapter_sensitivity":
            backbone_train_end = int(train_end * 0.8)
            validation_midpoint = train_end + (val_end - train_end) // 2
            holdout_length = train_end - backbone_train_end
            intervals = {
                "backbone_train": (0, backbone_train_end),
                "in_sample_matched": (
                    backbone_train_end - holdout_length, backbone_train_end
                ),
                "calibration_holdout": (backbone_train_end, train_end),
                "backbone_validation": (train_end, validation_midpoint),
                "adapter_validation": (validation_midpoint, val_end),
                "test": (val_end, self.num_time_steps),
            }
            default_role = {
                "train": "backbone_train",
                "val": "backbone_validation",
                "test": "test",
            }[flag]
            self.split_role = split_role or default_role
            if self.split_role not in intervals:
                raise ValueError(
                    f"Unsupported adapter-sensitivity split role: {self.split_role!r}."
                )
            self.split_start, self.split_end = intervals[self.split_role]
            statistics_end = backbone_train_end
        elif profile == "default":
            if split_role is not None:
                raise ValueError("split_role requires m3_split_profile=adapter_sensitivity.")
            self.split_role = flag
            self.split_start, self.split_end = {
                "train": (0, train_end),
                "val": (train_end, val_end),
                "test": (val_end, self.num_time_steps),
            }[flag]
            statistics_end = train_end
        else:
            raise ValueError(f"Unsupported M3 split profile: {profile!r}.")
        if self.split_end - self.split_start < self.seq_len + self.pred_len:
            raise ValueError(
                f"{self.split_role} split is too short for seq_len={self.seq_len} and "
                f"pred_len={self.pred_len}."
            )

        if self.scale:
            mean, std = self._state_stats(statistics_end)
            mean = mean[self.variable_indices]
            std = std[self.variable_indices]
        else:
            mean = np.zeros(self.num_features, dtype=np.float32)
            std = np.ones(self.num_features, dtype=np.float32)
        self.state_mean = mean
        self.state_std = std
        self.mean_out = mean.tolist()
        self.std_out = std.tolist()
        self.scaler = _RepeatedFeatureScaler(mean, std, self.num_stations)
        self.statistics_start = 0
        self.statistics_end = statistics_end

    def _calendar_marks(self, time_path):
        cache_key = os.path.realpath(time_path)
        cached = self._calendar_cache.get(cache_key)
        if cached is not None:
            return cached
        dates = pd.DatetimeIndex(self.times.astype("datetime64[ns]"))
        marks = np.stack(
            (
                (dates.month.to_numpy() - 1) / 12.0,
                (dates.day.to_numpy() - 1) / 31.0,
                dates.hour.to_numpy() / 24.0,
                (dates.dayofyear.to_numpy() - 1) / 366.0,
            ),
            axis=-1,
        ).astype(np.float32)
        self._calendar_cache[cache_key] = marks
        return marks

    def _state_stats(self, train_end, chunk_size=256):
        cache_key = (os.path.realpath(self.data_path), train_end)
        cached = self._stats_cache.get(cache_key)
        if cached is not None:
            return cached
        variables = len(M3_FRANCE_STATE_NAMES)
        total = np.zeros(variables, dtype=np.float64)
        total_sq = np.zeros(variables, dtype=np.float64)
        count = np.zeros(variables, dtype=np.float64)
        for start in range(0, train_end, chunk_size):
            stop = min(start + chunk_size, train_end)
            block = np.asarray(self.raw[:, :, start:stop], dtype=np.float32)
            state = m3_france_raw_to_state_np(np.moveaxis(block, -1, 0))
            finite = np.isfinite(state)
            values = np.where(finite, state, 0.0)
            total += values.sum(axis=(0, 1), dtype=np.float64)
            total_sq += np.square(values).sum(axis=(0, 1), dtype=np.float64)
            count += finite.sum(axis=(0, 1), dtype=np.float64)
        if not np.all(count == train_end * self.num_stations):
            raise ValueError("M3 France training states contain non-finite values.")
        mean = total / count
        variance = total_sq / count - np.square(mean)
        std = np.sqrt(np.maximum(variance, 1e-12))
        std[std < 1e-6] = 1.0
        result = mean.astype(np.float32), std.astype(np.float32)
        self._stats_cache[cache_key] = result
        return result

    def _state_window(self, start, length):
        block = np.asarray(
            self.raw[:, :, start:start + length], dtype=np.float32
        )
        state = m3_france_raw_to_state_np(np.moveaxis(block, -1, 0))
        state = state[..., self.variable_indices]
        if self.scale:
            state = (
                (state - self.state_mean.reshape(1, 1, -1))
                / self.state_std.reshape(1, 1, -1)
            )
        return state

    def __getitem__(self, index):
        s_begin = self.split_start + int(index)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_length = self.label_len + self.pred_len
        return (
            self._state_window(s_begin, self.seq_len),
            self._state_window(r_begin, r_length),
            self.calendar_marks[s_begin:s_end],
            self.calendar_marks[r_begin:r_begin + r_length],
            self.forecast_hour_offset + s_end,
        )

    def __len__(self):
        return self.split_end - self.split_start - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M3Europe(Dataset_M3France):
    """M3 Europe data with the same schema and preprocessing as M3 France."""

    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        data_path="m3_europe_europe_q0q1_weather5_1h_2017_2021.npy",
        scale=True,
        freq="h",
        split_role=None,
    ):
        super().__init__(
            args=args,
            root_path=root_path,
            flag=flag,
            size=size,
            data_path=data_path,
            scale=scale,
            freq=freq,
            split_role=split_role,
        )


class Dataset_M3Global(Dataset_M3France):
    """M3 Global data with the same schema and preprocessing as M3 France."""

    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        data_path="m3_global_global_q0_weather5_1h_2017_2021.npy",
        scale=True,
        freq="h",
        split_role=None,
    ):
        super().__init__(
            args=args,
            root_path=root_path,
            flag=flag,
            size=size,
            data_path=data_path,
            scale=scale,
            freq=freq,
            split_role=split_role,
        )
