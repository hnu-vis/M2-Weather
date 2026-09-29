# Long-term forecasting model catalog

This repository exposes both supervised forecasting and pretrained
inference through the single Time-Series-Library task name
`long_term_forecast`. There is no separate zero-shot experiment directory.
`models/<Model>.py` provides the unified entry and paper-specific components are
flat modules in `layers/`.

| Model | Long-term mode | Implementation | Source snapshot |
|---|---|---|---|
| Autoformer | supervised | `models/Autoformer.py` | `51c7d41` |
| CDPNet | supervised spatial | `models/CDPNet.py`, `layers/CDPNet_*.py` | `c0afafd` |
| Corrformer | supervised spatial | `models/Corrformer.py`, `layers/Corrformer_*.py` | `e3e0c6d` |
| DLinear | supervised | `models/DLinear.py` | `0c11366` |
| DUET | supervised | `models/DUET.py`, `layers/DUET_*.py` | `dcc6e67` |
| EasyST | supervised spatial, frozen-teacher distillation | `models/EasyST.py` | M3-native CIKM 2024 integration |
| HiSTGNN | supervised graph | `models/HiSTGNN.py`, `layers/HiSTGNN_Layer.py` | `7c32ee1` |
| MIGN | supervised graph | `models/MIGN.py`, `layers/MIGN_*.py` | `4398375` |
| Moirai | pretrained probabilistic inference | `models/Moirai.py` | `cfd46d4` |
| PatchTST | supervised | `models/PatchTST.py` | `204c21e` |
| STELLA | supervised spatial | `models/STELLA.py`, `layers/STELLA_*.py` | `cdb64e2` |
| Sundial | pretrained inference | `models/Sundial.py` | `3ef03b8` |
| TQNet | supervised | `models/TQNet.py`, `layers/TQNet.py` | `15e19cb` |
| Time-MoE | pretrained inference | `models/TimeMoE.py` | `915bfda` |
| Timer | Hugging Face zero-shot inference | `models/Timer.py` | `70077a7` |
| Timer-XL | OpenLTM zero-shot and pretrained continuation training | `models/TimerXL.py`, `layers/TimerXL_*.py` | `0b30050` |
| iTransformer | supervised | `models/iTransformer.py` | `c2426e6` |

## Unified task semantics

Moirai, Sundial, Time-MoE and Timer are inference-only entries. Run
them with `--task_name long_term_forecast --is_training 0`; `run.py` skips the
local TSLib checkpoint load because these models obtain their model/pipeline
externally. Moirai, Sundial, Time-MoE and Timer accept configurable pretrained model IDs.
Timer loads `thuml/timer-base-84m` through Hugging Face by default and applies
the official S3 univariate generation independently to each input variable.
Timer-XL uses the OpenLTM implementation and strictly loads
`pretrained_checkpoints/Timer-XL/checkpoint.pth` in both modes. With
`--is_training 0` it performs zero-shot rolling prediction; with
`--is_training 1` it continues training all parameters from those pretrained
weights using OpenLTM's univariate autoregressive token windows.

All remaining models use the supervised long-term experiment. Only forecast
constructors, heads and forward paths are retained in former TSLib multi-task
models; their long-term topology and equations are unchanged.

## Input contracts

- Every Dataset sample uses `x: [L,N,V]`, `y: [label+pred,N,V]`, where `N` is
  station count and `V` is meteorological-variable count. DataLoader adds `B`.
- Optional calendar features use `[L,F]` (or `[L,0]` when unavailable).
- Every sample also returns scalar `forecast_start`, the absolute sequence
  position of its first predicted point.
- Each model receives `[B,L,N,V]` and adapts it to its official native input
  contract internally, then restores dense predictions to `[B,P,N,V]` before
  the experiment loss.

| Scope | Models | Native value layout |
|---|---|---|
| multi-station, single-variable | CDPNet, Corrformer, EasyST, MIGN, STELLA | CDPNet/EasyST/MIGN/STELLA: `[B*V,L,N]`; Corrformer: `[B,L,N*V]`, restored internally to `V` input features per station |
| single-station, multi-variable | Timer-XL, Autoformer, iTransformer, DUET, TQNet, Moirai | `[B*N,L,V]` |
| single-station, single-variable | Time-MoE, Sundial, Timer, DLinear, PatchTST | Time-MoE/Sundial/Timer/PatchTST: `[B*N*V,L]`; DLinear: `[B*N,L,V]` |
| multi-station, multi-variable | HiSTGNN | `[B,L,N,V]` |

The categories describe explicit relationship modelling, not accepted input
channels. Corrformer accepts `V>1` exactly as its official implementation does:
`V` is projected as the feature channel of each station, while its explicit
cross-correlation operates over stations. It therefore remains a
multi-station, single-variable-relation model; value embedding alone is not
explicit variable-relation modelling.

- MIGN loads graph topology and spherical-harmonic metadata from a static DGL
  template, then writes each dense single-variable station tensor into a cloned
  graph internally. The Dataset therefore remains model-independent.

DUET exposes its official load-balancing loss through `auxiliary_loss`; the
long-term experiment adds it with `--aux_loss_weight`.


## Official training compatibility

`exp/training_strategies.py` selects the upstream objective, optimizer,
scheduler, clipping and early-stopping behavior while keeping the common
`exp/exp_long_term_forecasting.py` entry.

| Models | Configured training behavior |
|---|---|
| Autoformer, DLinear, Corrformer, iTransformer | Adam, MSE, TSLib epoch learning-rate adjustment and early stopping |
| PatchTST, TQNet | Adam, MSE and upstream OneCycleLR construction; per-batch stepping for `--lradj TST` |
| TQNet | The model adapter derives `cycle_index` from `forecast_start`, without timestamps |
| DUET | Adam, Huber loss with delta 0.5, `type3` schedule and load-balancing auxiliary loss |
| CDPNet | Adam with 1e-4 weight decay, standardized-space L1, epoch-0 direct target, later autoregressive sequence target, and early stopping after epoch 0 |
| Timer-XL | Strict pretrained initialization, OpenLTM univariate autoregressive token targets, Adam, MSE, and CosineAnnealingLR (`T_max defaults to train_epochs, eta_min=1e-8`) |
| STELLA | Standardized-space masked MAE, Adam with 5e-4 weight decay, MultiStepLR and max-norm 5 clipping |
| EasyST | MSE plus frozen-STELLA teacher-bounded regression and variational information-bottleneck losses; Adam and max-norm 5 clipping |
| HiSTGNN | Standardized-space masked MAE, Adam with 1e-4 weight decay and max-norm 5 clipping |
| MIGN | Dense labels with graph-native internal computation, AdamW and epoch-wise CosineAnnealingLR |

All models consume the same five-field Dataset batch. TQNet derives phase from window position.
CDPNet, Corrformer and STELLA load the same `[N,3]` latitude/longitude/altitude
array through `--station_coords_path` (default: `<root_path>/station_coords.npy`).
CDPNet accepts `--grid_lat_min`, `--grid_lat_max`, `--grid_lon_min` and
`--grid_lon_max`; omitted bounds are inferred from the coordinate file.
MIGN uses `--mign_config` for its architecture and
`--mign_graph_template` (or the first generated `.bin` below `data_dir`) for
static graph/Healpix metadata; only dense values come from each Dataset batch.
The four pretrained entries remain inference-only and continue to use
`long_term_forecast`. Timer-XL supports both zero-shot inference and continued
training through that same task.
