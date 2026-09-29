import torch


def make_grid(input):
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W, device=input.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=input.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    return grid

