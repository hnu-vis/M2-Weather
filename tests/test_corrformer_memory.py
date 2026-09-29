import importlib.util
from pathlib import Path
import unittest
import torch
from layers.Corrformer_Correlation import AutoCorrelation, CrossCorrelation
spec=importlib.util.spec_from_file_location('reference',Path(__file__).with_name('reference_corrformer_correlation.py'))
reference=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

class CorrelationTests(unittest.TestCase):
    def compare(self, old, new, shape, cross_shape=None):
        old.double();new.double();new.load_state_dict(old.state_dict())
        torch.manual_seed(31)
        original=[torch.randn(*(cross_shape if i else shape),dtype=torch.float64,requires_grad=True) for i in range(3)] if cross_shape else [torch.randn(*shape,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        optimized=[x.detach().clone().requires_grad_() for x in original]
        y0=old(*original,None)[0]; y1=new(*optimized,None)[0]
        torch.testing.assert_close(y0,y1,rtol=1e-10,atol=1e-10)
        y0.square().sum().backward();y1.square().sum().backward()
        for a,b in zip(original,optimized):torch.testing.assert_close(a.grad,b.grad,rtol=1e-9,atol=1e-9)
        for a,b in zip(old.parameters(),new.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=1e-9,atol=1e-9)
    def test_temporal_forward_and_backward(self):
        self.compare(reference.AutoCorrelation(factor=1),AutoCorrelation(factor=1),(2,12,2,3))
    def test_spatial_forward_and_backward(self):
        for factor in (1,2):
            self.compare(reference.CrossCorrelation(torch.nn.Conv1d(36,36,1),factor=factor),CrossCorrelation(torch.nn.Conv1d(36,36,1),factor=factor),(2,6,12,2,3))
    def test_cross_different_sequence_lengths(self):
        self.compare(reference.CrossCorrelation(torch.nn.Conv1d(36,36,1)),CrossCorrelation(torch.nn.Conv1d(36,36,1)),(2,6,12,2,3),(2,6,8,2,3))
if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
