"""Content-validated, batch-size-independent FP32 TimeMoE test predictions.

Each chunk is published through an atomic JSON commit after its NPY is durable.
Uncommitted chunks are ignored. Readers never unpickle cached data.
"""
from pathlib import Path
import hashlib
import inspect
import json
import os
import uuid

import numpy as np
import torch


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


class PredictionCache:
    def __init__(self, directory, identity):
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.directory = Path(directory) / key
        self.directory.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        self.hits = self.misses = 0
        self.loaded_name = None
        self.loaded = None
        for manifest in self.directory.glob('*.json'):
            try:
                info = json.loads(manifest.read_text())
                if info['identity'] != identity:
                    continue
                for row, sample in enumerate(info['samples']):
                    self.entries[int(sample['start'])] = (info['file'], row, sample)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        self.identity = identity
        print(f'TIMEMOE_CACHE_OPEN path={self.directory} windows={len(self.entries)}', flush=True)

    def get(self, batch):
        inputs, starts = _numpy(batch[0]), _numpy(batch[4]).reshape(-1)
        predictions = []
        try:
            for x, start in zip(inputs, starts):
                filename, row, sample = self.entries[int(start)]
                if sample['input_sha256'] != _digest(x):
                    raise ValueError('input mismatch')
                if filename != self.loaded_name:
                    self.loaded = np.load(self.directory / filename, mmap_mode='r', allow_pickle=False)
                    self.loaded_name = filename
                pred = self.loaded[row]
                if pred.dtype != np.float32 or list(pred.shape) != sample['shape'] or _digest(pred) != sample['prediction_sha256']:
                    raise ValueError('prediction mismatch')
                predictions.append(np.array(pred, copy=True))
            result = np.stack(predictions)
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            self.loaded_name = self.loaded = None
            self.misses += len(starts)
            return None
        self.hits += len(starts)
        return result

    def put(self, batch, prediction):
        prediction = np.asarray(_numpy(prediction), dtype=np.float32)
        inputs, starts = _numpy(batch[0]), _numpy(batch[4]).reshape(-1)
        if prediction.ndim != 4 or len(prediction) != len(starts) or len(inputs) != len(starts):
            raise ValueError('Cache expects matching [B,H,N,V] predictions and window IDs')
        filename = uuid.uuid4().hex + '.npy'
        path = self.directory / filename
        samples = [dict(start=int(start), input_sha256=_digest(x),
                        prediction_sha256=_digest(pred), shape=list(pred.shape))
                   for start, x, pred in zip(starts, inputs, prediction)]
        with path.open('xb') as handle:
            np.save(handle, prediction, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        info = dict(identity=self.identity, file=filename, samples=samples)
        temporary = path.with_suffix('.json.tmp')
        with temporary.open('x') as handle:
            json.dump(info, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path.with_suffix('.json'))
        for row, sample in enumerate(samples):
            self.entries[sample['start']] = (filename, row, sample)

    def report(self):
        print(f'TIMEMOE_CACHE_SUMMARY hit_windows={self.hits} miss_windows={self.misses} path={self.directory}', flush=True)


def open_test_cache(args, dataset, backbone):
    name = getattr(args, 'backbone_model', None) if args.model == 'InteractionAdapter' else args.model
    if (name != 'TimeMoE' or getattr(args, 'features', 'M') != 'M'
            or os.environ.get('TIMEMOE_PREDICTION_CACHE', '1') == '0'):
        return None
    # Hash actual weights, not only a mutable model repository name/revision.
    weights = hashlib.sha256()
    for name, value in sorted(backbone.state_dict().items()):
        weights.update(name.encode())
        tensor = value.detach().cpu().contiguous()
        weights.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        weights.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    data_path = Path(dataset.data_path).resolve()
    stat = data_path.stat()
    source_paths = [Path(__file__), Path(inspect.getfile(type(backbone))),
                    Path(inspect.getfile(type(backbone.model))),
                    Path(__file__).parent / 'transformers_compat.py',
                    Path(inspect.getfile(type(dataset)))]
    # Include generation implementation as well as model forward code.
    generation_path = source_paths[2].with_name('ts_generation_mixin.py')
    if generation_path.exists():
        source_paths.append(generation_path)
    identity = dict(version=1, weights=weights.hexdigest(),
                    sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                    config=backbone.model.config.to_dict(),
                    data=[str(data_path), stat.st_size, stat.st_mtime_ns],
                    split=[int(dataset.split_start), int(dataset.split_end)],
                    seq_len=int(args.seq_len), pred_len=int(args.pred_len),
                    mean=_digest(dataset.state_mean), std=_digest(dataset.state_std),
                    amp=bool(args.use_amp), chunk_size=int(backbone.inference_chunk_size),
                    torch_version=str(torch.__version__),
                    matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32)
    # Canonicalize configuration tuples and JSON keys before comparison.
    identity = json.loads(json.dumps(identity, sort_keys=True))
    directory = os.environ.get('TIMEMOE_PREDICTION_CACHE_DIR',
        str(Path(__file__).resolve().parents[1] / 'experiment_records' / 'prediction_cache' / 'TimeMoE'))
    return PredictionCache(directory, identity)
