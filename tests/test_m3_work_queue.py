import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
import unittest
from unittest.mock import patch
import schedule_m3_work_queue as q

class QueueTests(unittest.TestCase):
    def test_order_and_deterministic_seeds(self):
        jobs = q.jobs()
        self.assertEqual({j.seed for j in jobs if j.model in ('TimeMoE','Timer')}, {2024})
        first_hist = next(i for i,j in enumerate(jobs) if j.model=='HiSTGNN')
        self.assertEqual(jobs[first_hist].dataset,'Europe')
        self.assertTrue(any(j.model!='HiSTGNN' for j in jobs[first_hist + 1:]))
    def test_no_duplicate_and_teacher_dependency(self):
        job=q.Job('Global','EasyST',2024)
        with patch.object(q,'done',return_value=False):
            self.assertFalse(q.ready(job,set(),None,set(),{}))
            other=q.Job('Global','TimeMoE',2024)
            self.assertFalse(q.ready(other,{other.key},None,set(),{}))
            self.assertFalse(q.ready(other,set(),None,{other.key},{}))
            self.assertTrue(q.ready(other,set(),None,set(),{}))
    def test_legacy_exclusion(self):
        legacy=dict(chain={},model='Global:TimerXL')
        with patch.object(q,'done',return_value=False),patch.object(q,'alive',return_value=True):
            self.assertFalse(q.ready(q.Job('Global','TimerXL',2024),{},legacy,{},{}))
            self.assertTrue(q.ready(q.Job('Global','TimeMoE',2024),{},legacy,{},{}))
    def test_failed_job_does_not_block_other_work(self):
        failed=q.Job('Global','Corrformer',2024)
        next_job=q.Job('Global','Moirai',2024)
        with patch.object(q,'done',return_value=False):
            self.assertFalse(q.ready(failed,{},None,{}, {failed.key: {}}))
            self.assertTrue(q.ready(next_job,{},None,{}, {failed.key: {}}))
    def test_corrformer_and_histgnn_are_ordinary_ready_jobs(self):
        corr=q.Job('Global','Corrformer',2024)
        other=q.Job('Global','TimeMoE',2024)
        hist=q.Job('Global','HiSTGNN',2024)
        with patch.object(q,'jobs',return_value=[corr,other,hist]):
            with patch.object(q,'done',side_effect=lambda j: False):
                self.assertTrue(q.ready(corr,{},None,{},{}))
                self.assertTrue(q.ready(hist,{},None,{},{}))
            with patch.object(q,'done',side_effect=lambda j: j==other):
                self.assertTrue(q.ready(corr,{other.key},None,{},{}))
                self.assertFalse(q.ready(corr,{corr.key},None,{},{}))
                self.assertTrue(q.ready(corr,{hist.key},None,{hist.key},{}))
    def test_corrformer_runs_preflight_wrapper(self):
        cmd,env=q.command(q.Job('Global','Corrformer',2024),3)
        self.assertTrue(cmd[-1].endswith('run_m3_global_corrformer_checked.sh'))
    def test_single_seed_configuration(self):
        cmd,env=q.command(q.Job('Global','Corrformer',2025),4)
        self.assertEqual(env['SEEDS'],'2025')
        self.assertEqual(env['GPU_IDS'],env['ADAPTER_GPUS'])
        self.assertEqual(env['GPU_IDS'],'4')
        self.assertEqual(env['TRAIN_STRIDE'],'6')
    def test_occupied_gpu_not_dispatched(self):
        with patch.object(q.subprocess,'check_output',return_value='0, 20000, 0\n1, 1, 0\n2, 1, 0\n'):
            self.assertEqual(q.idle_gpus({0,1,2},{1}),[2])

if __name__=='__main__':unittest.main()
