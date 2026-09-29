import copy
import unittest

import torch

from models.HiSTGNN import HierarchicalNet
from layers.HiSTGNN_Layer import nconv


def make_model(vectorized):
    return HierarchicalNet(
        seq_length=12,
        n_var=4,
        n_stat=3,
        var_dim=5,
        stat_dim=6,
        device="cpu",
        tanhalpha=3,
        conv_channels=8,
        gcn_depth=2,
        residual_channels=8,
        in_dim=2,
        dropout=0.0,
        end_channels=16,
        out_dim=5,
        propalpha=0.05,
        predefined_A=True,
        dilation_exponential=1,
        layers=2,
        skip_channels=8,
        gcn_true=True,
        gat_true=False,
        hier_true=True,
        DIL_true=False,
        fusion="AvgPool",
        diffusion="GatedCopy",
        A_type="uni-directed",
        conv_k_size=(1, 1, 1),
        kernel_size=6,
        vectorized=vectorized,
    )


class HiSTGNNVectorizedTest(unittest.TestCase):
    def test_gemm_graph_contraction_matches_einsum(self):
        torch.manual_seed(3)
        values = torch.randn(2, 5, 7, 11, dtype=torch.double,
                             requires_grad=True)
        adjacency = torch.randn(7, 7, dtype=torch.double, requires_grad=True)
        expected = torch.einsum("ncwl,vw->ncvl", values, adjacency)
        actual = nconv()(values, adjacency)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_forward_and_backward_match_reference(self):
        torch.manual_seed(7)
        reference = make_model(vectorized=False).double().eval()
        vectorized = copy.deepcopy(reference)
        vectorized.vectorized = True

        reference_input = torch.randn(2, 2, 3, 4, 12, dtype=torch.double,
                                      requires_grad=True)
        vectorized_input = reference_input.detach().clone().requires_grad_(True)
        reference_output = reference(reference_input)
        vectorized_output = vectorized(vectorized_input)
        torch.testing.assert_close(
            vectorized_output, reference_output, rtol=1e-9, atol=1e-10
        )

        reference_output.square().sum().backward()
        vectorized_output.square().sum().backward()
        torch.testing.assert_close(
            vectorized_input.grad, reference_input.grad, rtol=2e-8, atol=1e-9
        )
        reference_parameters = dict(reference.named_parameters())
        vectorized_parameters = dict(vectorized.named_parameters())
        self.assertEqual(reference_parameters.keys(), vectorized_parameters.keys())
        for name in reference_parameters:
            torch.testing.assert_close(
                vectorized_parameters[name].grad,
                reference_parameters[name].grad,
                rtol=2e-8,
                atol=1e-9,
                msg=lambda message, name=name: f"{name}: {message}",
            )


if __name__ == "__main__":
    unittest.main()
