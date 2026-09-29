import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        # self.linear_time = nn.Linear(48, 48)
        # self.linear_space = nn.Linear(3850, 3850)
        # self.linear_feature = nn.Linear(128, 128)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """
        # input_data = self.linear_feature(input_data)
        # hidden = self.linear_time(input_data.transpose(2, 3)).transpose(2, 3)
        # # hidden = self.linear_space(hidden.transpose(1, 3)).transpose(1, 3)
        # hidden = self.drop(self.act(hidden))
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden
