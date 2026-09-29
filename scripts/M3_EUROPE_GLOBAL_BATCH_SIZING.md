# M3 Europe / Global batch-size policy

The France scripts use 176 stations. Europe has 983 stations and Global has 2504.
Defaults below keep `batch_size * station_count` close to the measured France workload
for models whose station dimension is folded into the sample axis. Explicit spatial
models use the same linear rule with conservative rounding. The scheduler retries CUDA
OOM failures with batch sizes divided by 2 on each retry.

| Model | France (176) | Europe (983) | Global (2504) |
|---|---:|---:|---:|
| DLinear | 2048 | 384 | 144 |
| TQNet | 1024 | 192 | 72 |
| xPatch | 1024 | 192 | 72 |
| STELLA | 1024 | 192 | 64 |
| EasyST | 512 | 96 | 32 |
| S2Transformer | 384 | 64 | 24 |
| CDPNet | 256 | 48 | 16 |
| DUET | 256 | 48 | 16 |
| PatchTST | 128 | 24 | 8 |
| iTransformer | 128 | 24 | 8 |
| Timer | 128 | 24 | 8 |
| TimerXL | 64 | 12 | 4 |
| HiSTGNN | 64 | 20 | 10 (AMP/vectorized; eval 21) |
| Corrformer | 8 | 1 | 1 |
| TimeMoE | 4 | 1 | 1 |
| Moirai | 8 | 1 | 1 |

S2Transformer geographic group counts are scaled from France `16 4` to Europe
`96 24` and Global `224 56` so local group width remains bounded. Corrformer's
`node_list` must multiply exactly to the station count: Europe uses `983` because
983 is prime; Global uses `8 313`.

Run all baselines three times:

```bash
bash scripts/run_m3_europe_repeated.sh
bash scripts/run_m3_global_repeated.sh
```
