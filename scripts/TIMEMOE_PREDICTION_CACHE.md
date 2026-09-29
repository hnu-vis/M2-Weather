# TimeMoE test prediction reuse

TimeMoE baseline test evaluation automatically writes normalized FP32 predictions.
The closed-form Adapter test reuses them; train/validation fitting is unchanged.
The existing running scheduler needs no restart: future Python jobs load this code.

Default storage: `experiment_records/prediction_cache/TimeMoE/<identity>/`.
Global test predictions require approximately 37.6 GB plus small manifests.
Override storage with `TIMEMOE_PREDICTION_CACHE_DIR`; disable using
`TIMEMOE_PREDICTION_CACHE=0` in the job environment.

Identity covers actual weights, model/configuration/source code, data file identity,
split, normalization, context/horizon, chunk size and numerical settings.
Each window also checks normalized input and prediction SHA-256 hashes.
Window IDs allow different batch sizes and order. Committed chunks survive a
restart; incomplete/corrupt/mismatched entries are recomputed. NPY is read with
`allow_pickle=False`. FP32 predictions are stored before inverse transformation.

Logs: `TIMEMOE_CACHE_OPEN` and `TIMEMOE_CACHE_SUMMARY` report location and hits/misses.
Only TimeMoE with `features=M` uses this cache. No GPU scheduling or reduced-precision model changes are involved. TimeMoE
baseline matmul TF32 is disabled to match the existing Adapter FP32 inference
setting; other models retain their existing settings. Historical baseline
metrics may differ slightly because of this alignment. Deleting the cache after its consumers finish only removes
this acceleration; it does not remove model results.

CPU regression checks:

```sh
python -m unittest discover -s tests -p test_timemoe_prediction_cache.py -v
```
