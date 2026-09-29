from types import SimpleNamespace

import numpy as np

from data_provider.data_loader import Dataset_M3France


def _dataset(role):
    args = SimpleNamespace(
        m3_split_profile="adapter_sensitivity",
        state_variables=None,
    )
    return Dataset_M3France(
        args=args,
        root_path="dataset/M3/France",
        data_path="m3_france_france_q0q1_weather5_1h_2017_2021.npy",
        flag="test" if role == "test" else "train",
        size=[48, 0, 72],
        split_role=role,
    )


def test_sensitivity_intervals_and_counts_are_leakage_safe():
    roles = [
        "backbone_train", "calibration_holdout", "backbone_validation",
        "adapter_validation", "test",
    ]
    datasets = {role: _dataset(role) for role in roles}
    assert [len(datasets[role]) for role in roles] == [20916, 5140, 2072, 2072, 13029]
    for left, right in zip(roles, roles[1:]):
        assert datasets[left].split_end == datasets[right].split_start
        last_left_label_index = datasets[left].split_end - 1
        first_right_input_index = datasets[right].split_start
        assert last_left_label_index < first_right_input_index


def test_matched_block_and_normalizer_are_shared():
    train = _dataset("backbone_train")
    matched = _dataset("in_sample_matched")
    held = _dataset("calibration_holdout")
    adapter_val = _dataset("adapter_validation")
    assert len(matched) == len(held) == 5140
    assert matched.split_end == train.split_end
    assert matched.split_end - matched.split_start == held.split_end - held.split_start
    assert all(ds.statistics_end == train.split_end for ds in (matched, held, adapter_val))
    for dataset in (matched, held, adapter_val):
        np.testing.assert_array_equal(dataset.state_mean, train.state_mean)
        np.testing.assert_array_equal(dataset.state_std, train.state_std)

