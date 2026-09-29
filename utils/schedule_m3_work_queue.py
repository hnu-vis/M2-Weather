"""One GPU per model/seed baseline->Adapter work queue."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from write_m3_progress import DATASETS, MODELS, MODES, any_match, running_jobs

ROOT = Path(__file__).resolve().parents[1]
MAPS = {
    'Europe': 'DLinear=6;TQNet=2;STELLA=8;DUET=3;iTransformer=2;Timer=32;TimeMoE=24;Moirai=16',
    'Global': 'DLinear=6;TQNet=2;STELLA=8;DUET=3;iTransformer=2;Timer=32;TimeMoE=24;Moirai=8',
}

@dataclass(frozen=True)
class Job:
    dataset: str
    model: str
    seed: int

    @property
    def key(self):
        return f'{self.dataset}:{self.model}:{self.seed}'


def process_token(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] not in ('Z', 'X') else None
    except (OSError, IndexError):
        return None


def alive(record):
    return record.get('token') is not None and process_token(record['pid']) == record['token']


def done(job, baseline_only=False):
    data_name, stem = DATASETS[job.dataset]
    root = ROOT / 'experiment_records' / stem
    baseline = any_match(root / 'baselines/results', f'*_{job.model}_{data_name}_*seed{job.seed}_0/metrics.npy')
    adapter = (root / 'adapters' / f'seed{job.seed}' / 'results' / f'{job.model}_{MODES[job.model]}_closed_form/metrics.npy').is_file()
    return baseline if baseline_only else baseline and adapter


def jobs():
    result = [Job(ds, model, seed) for ds in DATASETS for model in MODELS
              for seed in ((2024,) if model in ('Timer', 'TimeMoE') else (2024, 2025, 2026))]
    # Prefer long jobs first; every model, including HiSTGNN, otherwise follows
    # the same readiness and one-GPU scheduling rules.
    order = {'TimeMoE': 0, 'Corrformer': 1, 'Moirai': 2, 'Timer': 3}
    return sorted(
        result,
        key=lambda j: (order.get(j.model, 4), j.dataset, MODELS.index(j.model), j.seed),
    )


def ready(job, active_keys, legacy, live_keys, failed):
    if job.key in active_keys or job.key in live_keys or job.key in failed or done(job):
        return False
    if legacy and alive(legacy['chain']) and f'{job.dataset}:{job.model}' == legacy['model']:
        return False
    if job.dataset == 'Global' and job.model == 'Corrformer':
        # The checked worker gives each seed an exclusive GPU and validates the
        # CPU-offload implementation first. It may safely overlap with other
        # models now that saved activations no longer fill GPU memory.
        if legacy and alive(legacy['chain']):
            return False
    if job.model == 'EasyST' and not done(Job(job.dataset, 'STELLA', job.seed), baseline_only=True):
        return False
    return True


def command(job, gpu):
    env = os.environ.copy()
    # Do not inherit output paths/model filters from an old worker shell.
    for key in ('RECORD_ROOT', 'CHECKPOINT_ROOT', 'ADAPTER_ROOT', 'SUMMARY_ROOT',
                'BASELINE_LOG_ROOT', 'ADAPTER_LOG_ROOT', 'JOB_TMPDIR',
                'SEED_OVERRIDES', 'ADAPTER_SEED_OVERRIDES', 'CUDA_VISIBLE_DEVICES'):
        env.pop(key, None)
    env.update(RUN_VARIANT='fast_s6', TRAIN_STRIDE='6', SKIP_EPOCH_TEST='1',
               ADAPTER_FIT_STRIDE='6', REUSE_DETERMINISTIC='1',
               GPU_IDS=str(gpu), ADAPTER_GPUS=str(gpu), SEEDS=str(job.seed),
               MODELS=job.model, RUN_EXPERIMENTS='1', RUN_SUMMARY='0',
               PRECOMPLETED_ADAPTERS_FIRST='0', PREPASS_ONLY='0', DRY_RUN='0',
               BATCH_SIZE_MULTIPLIERS=MAPS[job.dataset],
               ADAPTER_BATCH_SIZE_MULTIPLIERS=MAPS[job.dataset],
               MAX_JOBS_PER_GPU='1', TSLIB_PYTHON=sys.executable)
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    script = ('run_m3_global_corrformer_checked.sh' if job.dataset == 'Global' and job.model == 'Corrformer'
              else f'run_m3_{job.dataset.lower()}_repeated.sh')
    return ['bash', str(ROOT / 'scripts' / script)], env


def idle_gpus(gpus, reserved):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True)
    return [int(fields[0]) for line in output.splitlines()
            if len(fields := [s.strip() for s in line.split(',')]) == 3
            and int(fields[0]) in gpus and int(fields[0]) not in reserved
            and int(fields[1]) < 1024 and int(fields[2]) < 10]


def log(message):
    print(f'{datetime.now().isoformat(timespec="seconds")} WORK_QUEUE {message}', flush=True)


def save(path, state):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--retry-global-corrformer', action='store_true')
    parser.add_argument('--gpus', default='0,1,2,3,4,5')
    parser.add_argument('--legacy-parent', type=int)
    parser.add_argument('--legacy-chain', type=int)
    parser.add_argument('--legacy-monitor', type=int)
    parser.add_argument('--legacy-gpus', default='0,1,2')
    parser.add_argument('--legacy-model', default='Global:TimerXL')
    args = parser.parse_args()
    queue = jobs()
    if args.dry_run:
        for job in queue:
            if not done(job):
                cmd, env = command(job, 3)
                print(json.dumps(dict(job=job.key, command=cmd, env={k: env[k] for k in
                    ('SEEDS', 'MODELS', 'GPU_IDS', 'ADAPTER_GPUS', 'RUN_VARIANT', 'TRAIN_STRIDE', 'REUSE_DETERMINISTIC')})))
        return
    directory = ROOT / 'logs/pipeline'
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'work_queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.set_inheritable(lock.fileno(), False)
    state_path = directory / 'work_queue_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(active={}, failed={}, legacy=None)
    if args.retry_global_corrformer:
        for key in list(state['failed']):
            if key.startswith('Global:Corrformer:'):
                state.setdefault('failure_history', []).append(dict(key=key, record=state['failed'].pop(key),
                    requeued_at=datetime.now().isoformat(), reason='FP32 CPU activation offload and allocation fix'))
                log(f'REQUEUE_DEFERRED job={key} barrier=all_other_non_HiSTGNN_complete')
    if args.legacy_parent and not state.get('legacy'):
        def record(pid):
            return dict(pid=pid, token=process_token(pid))
        parent = record(args.legacy_parent)
        if not alive(parent):
            raise RuntimeError('Legacy scheduler is not alive')
        status = Path(f'/proc/{args.legacy_parent}/status').read_text()
        if 'State:\tT' not in status:
            raise RuntimeError('Legacy scheduler must be stopped before takeover')
        state['legacy'] = dict(parent=parent, chain=record(args.legacy_chain),
            monitor=record(args.legacy_monitor), model=args.legacy_model,
            gpus=[int(v) for v in args.legacy_gpus.split(',')])
    children = {}
    save(state_path, state)
    log('START adopted=' + str(state['legacy']))
    gpus = set(int(v) for v in args.gpus.split(','))
    while True:
        legacy = state.get('legacy')
        if legacy and not alive(legacy['chain']):
            # Only the stopped old scheduling parent and its progress monitor
            # are retired; the adopted model chain has already exited.
            for name in ('monitor', 'parent'):
                rec = legacy[name]
                if alive(rec):
                    os.kill(rec['pid'], signal.SIGKILL if name == 'parent' else signal.SIGTERM)
            log('LEGACY_RELEASE ' + legacy['model'])
            state['legacy'] = legacy = None
        for key, record in list(state['active'].items()):
            process = children.get(key)
            if process is not None:
                process.poll()
            if alive(record):
                continue
            job = Job(**record['job'])
            success = done(job)
            log(f'FINISH job={key} gpu={record["gpu"]} success={success}')
            if not success:
                state['failed'][key] = record
            del state['active'][key]
            children.pop(key, None)
        baselines, adapters = running_jobs()
        live_keys = {f'{ds.removeprefix("M3")}:{model}:{seed}' for ds, model, seed in baselines | adapters}
        reserved = {rec['gpu'] for rec in state['active'].values()}
        if legacy:
            reserved.update(legacy['gpus'])
        for gpu in idle_gpus(gpus, reserved):
            job = next((j for j in queue if ready(j, state['active'], legacy, live_keys, state['failed'])), None)
            if job is None:
                break
            cmd, env = command(job, gpu)
            path = directory / f'worker_{job.dataset}_{job.model}_{job.seed}.log'
            with path.open('ab') as output:
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            children[job.key] = process
            state['active'][job.key] = dict(pid=process.pid, token=process_token(process.pid),
                gpu=gpu, job=vars(job), log=str(path))
            save(state_path, state)
            log(f'DISPATCH job={job.key} gpu={gpu} pid={process.pid} log={path}')
        save(state_path, state)
        subprocess.run([sys.executable, str(ROOT / 'utils/write_m3_progress.py')], check=True)
        pending = [j for j in queue if not done(j)]
        if not pending and not state['active'] and not legacy:
            for ds in DATASETS:
                cmd, env = command(Job(ds, 'DLinear', 2024), 0)
                env.update(RUN_EXPERIMENTS='0', RUN_SUMMARY='1', SEEDS='2024,2025,2026')
                subprocess.run(cmd, env=env, cwd=ROOT, check=True)
            log('COMPLETE')
            return
        if pending and not state['active'] and not legacy and not live_keys and all(j.key in state['failed'] or (j.model == 'EasyST' and Job(j.dataset, 'STELLA', j.seed).key in state['failed']) for j in pending):
            log('FAILED remaining=' + ','.join(j.key for j in pending))
            raise SystemExit(1)
        time.sleep(5)

if __name__ == '__main__':
    main()
