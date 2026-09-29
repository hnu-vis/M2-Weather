# M3 dynamic GPU queue

Start/restart with `bash scripts/run_m3_work_queue.sh`. Do not start the old
`run_m3_europe_global_fast_s6.sh` concurrently. The new supervisor holds an
exclusive lock and recovers still-running worker PIDs from persisted state.

Each available GPU receives one dataset/model/seed baseline + Adapter chain.
Completed outputs are skipped. Timer/TimeMoE run only seed 2024 and retain the
existing deterministic result reuse. EasyST requires its teacher baseline.
Long non-HiSTGNN tasks are dispatched first. HiSTGNN can start when no ready
non-HiSTGNN task remains queued, even while other tasks run; Global HiSTGNN
is prioritized over Europe because of its longer projected duration.

Existing TimerXL was adopted intact on GPUs 0-2. Its old scheduling parent is
stopped, not its training workers. When that chain exits, the supervisor retires
the old parent/monitor and adds GPUs 0-2 to the independent pool. This one adopted
chain retains its original seed-group barrier until completion.

The pool checks availability every five seconds (plus status scan time), skips
GPUs with >=1 GiB allocated or utilization >=10%, and reserves GPUs throughout
worker startup and baseline->Adapter transitions. Loading, validation and I/O
can still produce transient low utilization; 100% utilization is not guaranteed.
Existing worker OOM retries remain enabled. Exhausted failures are recorded and
do not block other jobs; they are not marked completed or silently retried forever.

Logs: `logs/pipeline/work_queue.log`, `logs/pipeline/worker_*.log`.
State/failures: `logs/pipeline/work_queue_state.json`.
Progress: `logs/progress.csv` (failed jobs remain unfinished there; consult queue state).
For inspection only: `bash scripts/run_m3_work_queue.sh --dry-run`.

## Global Corrformer memory fix and deferred retry

Global Corrformer now waits until every OTHER non-HiSTGNN baseline and Adapter
chain is complete (including running worker cleanup). HiSTGNN is excluded from
this barrier. Historical Corrformer failures are archived when explicitly
requeued with `--retry-global-corrformer`; subsequent failures still stop that
seed rather than retry forever.

The checked worker first executes two full Global-shape FP32 Adam steps and
evaluation on its exclusively reserved GPU. Only a successful preflight permits
formal training. Reports: `logs/pipeline/corrformer_preflight_seed*.json`.
The fix replaces materialized repeated indices/values with circular broadcast
indices and offloads saved training activations to pinned CPU memory for >=2000
stations. Parameters, optimizer and compute stay on GPU; no station subsampling,
model-size reduction or mixed precision is introduced. Host memory and transfer
time increase; actual GPU peak is recorded by the deferred preflight.
