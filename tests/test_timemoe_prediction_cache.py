import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils.timemoe_prediction_cache import PredictionCache
from utils import fit_interaction_adapter as fitting


class Windows(Dataset):
    num_features = 2
    state_std = np.array([2., 3.])

    def __len__(self):
        return 5

    def __getitem__(self, i):
        return (torch.full((3, 2, 2), float(i)),
                torch.full((2, 2, 2), float(i + 1)),
                torch.zeros(3, 1), torch.zeros(2, 1), 100 + i)


def prediction(batch):
    return batch[0][:, :2] + 0.25


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = PredictionCache(self.tmp.name, {'weights': 'a'})

    def test_rebatch_and_reorder(self):
        for batch in DataLoader(Windows(), batch_size=3):
            self.cache.put(batch, prediction(batch))
        reader = PredictionCache(self.tmp.name, {'weights': 'a'})
        for batch in DataLoader(Windows(), batch_size=2, sampler=[4, 0, 3, 1, 2]):
            np.testing.assert_array_equal(reader.get(batch), prediction(batch).numpy())
        self.assertEqual(reader.hits, 5)

    def test_input_model_and_corruption_rejected(self):
        batch = next(iter(DataLoader(Windows(), batch_size=2)))
        np.save(self.cache.directory / 'uncommitted.npy', prediction(batch).numpy())
        self.assertIsNone(PredictionCache(self.tmp.name, {'weights': 'a'}).get(batch))
        self.cache.put(batch, prediction(batch))
        changed = list(batch)
        changed[0] = changed[0] + 1
        self.assertIsNone(self.cache.get(changed))
        self.assertIsNone(PredictionCache(self.tmp.name, {'weights': 'b'}).get(batch))
        name = self.cache.entries[100][0]
        arr = np.load(self.cache.directory / name, mmap_mode='r+')
        arr[0, 0, 0, 0] += 1
        arr.flush()
        self.assertIsNone(PredictionCache(self.tmp.name, {'weights': 'a'}).get(batch))

    def test_adapter_metrics_and_zero_forward_calls(self):
        dataset = Windows()
        args = SimpleNamespace(pred_len=2)
        wrapper = SimpleNamespace(backbone=object(), adapter=torch.nn.Identity())
        wrapper.adapter.num_variables = 2
        loader = DataLoader(dataset, batch_size=2)
        def forward(wrapper, batch, device, args):
            return prediction(batch), batch[1]
        with patch.object(fitting, '_backbone_prediction', side_effect=forward) as call:
            with patch('utils.timemoe_prediction_cache.open_test_cache', return_value=None):
                expected = fitting._evaluate(wrapper, dataset, loader, torch.device('cpu'), args)
            self.assertEqual(call.call_count, 3)
            for batch in DataLoader(dataset, batch_size=3):
                self.cache.put(batch, prediction(batch))
            reader = PredictionCache(self.tmp.name, {'weights': 'a'})
            call.reset_mock()
            with patch('utils.timemoe_prediction_cache.open_test_cache', return_value=reader):
                actual = fitting._evaluate(wrapper, dataset, loader, torch.device('cpu'), args)
            call.assert_not_called()
            self.assertEqual(actual, expected)
            self.assertEqual(reader.hits, 5)

    def test_baseline_adapter_identity_matches_and_weights_invalidate(self):
        from pathlib import Path
        from utils.timemoe_prediction_cache import open_test_cache
        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2))
                self.config = SimpleNamespace(to_dict=lambda: {'model_type': 'time_moe'})
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Inner()
                self.inference_chunk_size = 1024
        dataset = Windows()
        dataset.data_path = str(Path(self.tmp.name) / 'data.npy')
        np.save(dataset.data_path, np.zeros(1))
        dataset.state_mean = np.zeros(2)
        dataset.split_start, dataset.split_end = 0, 10
        args = SimpleNamespace(model='TimeMoE', seq_len=3, pred_len=2, use_amp=False)
        backbone = Backbone()
        with patch.dict('os.environ', {'TIMEMOE_PREDICTION_CACHE_DIR': self.tmp.name}):
            baseline = open_test_cache(args, dataset, backbone)
            args.model, args.backbone_model = 'InteractionAdapter', 'TimeMoE'
            adapter = open_test_cache(args, dataset, backbone)
            self.assertEqual(baseline.directory, adapter.directory)
            with torch.no_grad():
                backbone.model.weight.add_(1)
            changed = open_test_cache(args, dataset, backbone)
            self.assertNotEqual(baseline.directory, changed.directory)

    def test_baseline_produces_and_reuses_cache(self):
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        from pathlib import Path
        dataset = Windows()
        dataset.num_stations = 2
        experiment = object.__new__(Exp_Long_Term_Forecast)
        experiment.args = SimpleNamespace(model='TimeMoE', pred_len=2, use_dtw=False,
            inverse=False, test_results=self.tmp.name, results=self.tmp.name,
            result_file=str(Path(self.tmp.name) / 'results.txt'))
        experiment.model = torch.nn.Identity()
        experiment._get_data = lambda flag: (dataset, DataLoader(dataset, batch_size=3))
        experiment._select_criterion = lambda: None
        def forward(batch, *args, **kwargs):
            return prediction(batch).flatten(2), batch[1].flatten(2), None
        with patch.object(experiment, '_forward_batch', side_effect=forward) as call:
            with patch('utils.timemoe_prediction_cache.open_test_cache', return_value=self.cache):
                experiment.test('fresh')
            self.assertEqual(call.call_count, 2)
            call.reset_mock()
            reader = PredictionCache(self.tmp.name, {'weights': 'a'})
            experiment._get_data = lambda flag: (dataset, DataLoader(dataset, batch_size=2))
            with patch('utils.timemoe_prediction_cache.open_test_cache', return_value=reader):
                experiment.test('cached')
            call.assert_not_called()
        np.testing.assert_array_equal(np.load(Path(self.tmp.name) / 'fresh/metrics.npy'),
                                      np.load(Path(self.tmp.name) / 'cached/metrics.npy'))


if __name__ == '__main__':
    unittest.main()
