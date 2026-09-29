"""TSLib shape adapter for the STELLA architecture."""

import torch
import torch.nn as nn

from layers.STELLA_Arch import STELLA


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_nodes = getattr(configs, "num_nodes", 1)
        self.num_features = 1
        self.model = STELLA(
            num_nodes=self.num_nodes,
            num_features=self.num_features,
            input_len=configs.seq_len,
            d_model=configs.d_model,
            output_len=configs.pred_len,
            num_layer=getattr(configs, "e_layers", 2),
            if_rel=getattr(configs, "if_rel", True),
            res_conn=getattr(configs, "res_conn", True),
            root_path=configs.root_path,
            station_coords_path=getattr(configs, "station_coords_path", None),
            dropout=configs.dropout,
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.ndim != 4:
            raise ValueError(f"STELLA expects [B,L,N,V], got {tuple(x_enc.shape)}.")
        batch_size, length, num_stations, num_variables = x_enc.shape
        x_enc = x_enc.permute(0, 3, 1, 2).reshape(
            batch_size * num_variables, length, num_stations
        )
        if x_mark_enc is not None:
            if x_mark_enc.shape[-1] == 0:
                x_mark_enc = None
            else:
                x_mark_enc = x_mark_enc.repeat_interleave(num_variables, dim=0)
        if x_enc.shape[-1] != self.num_nodes:
            raise ValueError(
                f"STELLA was configured for {self.num_nodes} stations, got "
                f"{x_enc.shape[-1]}."
            )
        history = x_enc.unsqueeze(-1)

        if x_mark_enc is not None:
            marks = x_mark_enc.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
            history = torch.cat((history, marks), dim=-1)
        output = self.model(history)
        output = output.flatten(start_dim=2)
        return output.reshape(
            batch_size, num_variables, output.shape[1], num_stations
        ).permute(0, 2, 3, 1).contiguous()
