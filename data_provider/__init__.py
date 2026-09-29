"""Unified Dataset protocol for the station-weather benchmark.

Each ``Dataset.__getitem__`` returns exactly five arrays/tensors or scalars:

``x``: ``[input_time, station, variable]``
``y``: ``[label_time + prediction_time, station, variable]``
``x_mark``: ``[input_time, calendar_feature]``
``y_mark``: ``[label_time + prediction_time, calendar_feature]``
``forecast_start``: absolute integer position of the first predicted point

The DataLoader adds the batch axis. Dataset implementations expose
``num_stations`` and ``num_features`` and must not branch on model name.

Calendar marks may have zero features when unavailable. Scaled datasets also
expose per-variable statistics through their scaler.
"""
