# Adapter residual-fit source sensitivity

This experiment retrains each backbone on the first 80% of the original M3 training interval, selects its checkpoint on the first half of the original validation interval, and compares three Adapter coefficient-fit sources on that one frozen checkpoint. The original test interval is unchanged and is not touched until all ridge values have been selected.

## Reproduce

Run France first on six GPUs:

```bash
cd /root/autodl-tmp/M3_benchmark
/root/miniconda3/envs/tslib/bin/python -u \
  experiments/adapter_fit_source_sensitivity/run_sweep.py \
  --datasets France --gpus 0,1,2,3,4,5
```

Global is no longer part of the active experiment. The code path remains available only for a future explicit extension:

```bash
/root/miniconda3/envs/tslib/bin/python -u \
  experiments/adapter_fit_source_sensitivity/run_sweep.py \
  --datasets Global --gpus 0,1,2,3,4,5
```

The runner is resumable. A completed `summary.json` skips a job; interrupted backbone training resumes from `last_checkpoint.pth`.

Build the seed-level and mean/sample-standard-deviation tables:

```bash
/root/miniconda3/envs/tslib/bin/python \
  experiments/adapter_fit_source_sensitivity/aggregate.py
```

## Fixed design

- Datasets: France and Global; 48-hour input, 72-hour prediction, T/WS/RH/P.
- Backbones: PatchTST (both interaction branches), iTransformer (station branch), S²Transformer (variable branch).
- Seeds: 2024, 2025, 2026.
- Backbone optimizer/objective: Adam, prediction MSE, initial learning rate `1e-4`, `type1`; S²Transformer retains AMP, `weight_decay=1e-4`, and gradient clipping at 5.
- All formal training and Adapter fitting prediction origins use stride 1.
- Ridge candidates: `0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1`.
- In-sample matched uses the latest contiguous raw-time block inside backbone-train with exactly the same duration and number of valid origins as calibration-holdout.

## Outputs

Each `artifacts/adapter_fit_source_sensitivity/<Dataset>/<Model>/seed_<seed>/` contains:

- `checkpoints/.../checkpoint.pth`: best backbone selected by backbone-validation;
- `logs/backbone.log` and `logs/adapter.log`: full commands and progress;
- `status.json`: stage, device, wall-clock training and Adapter-analysis duration;
- `sensitivity/summary.json`: intervals, leakage checks, ridge trials, errors, coefficient norms, timing, and all test metrics;
- `sensitivity/metrics.csv`: compact per-seed table;
- `sensitivity/coefficients.pt`: selected and shared-ridge coefficients.

`artifacts/adapter_fit_source_sensitivity/aggregate/` contains the combined per-seed table, mean and sample-standard-deviation table, and matched-versus-held-out paired differences. The aggregation status explicitly lists missing runs and does not generate a paper conclusion while the requested runs are incomplete.
