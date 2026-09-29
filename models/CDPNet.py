import numpy as np
import sklearn
import sklearn.neighbors
import torch
import torch.nn as nn
from layers.CDPNet_MLP import MultiLayerPerceptron
from layers.CDPNet_Utils import make_grid
from utils.station_coordinates import load_station_coordinates


class Evolution(nn.Module):
    def __init__(self, configs, distances, neighbors, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size):
        super(Evolution, self).__init__()
        self.feat_in = configs.feat_in
        self.input_dim = configs.input_dim
        self.num_node = configs.num_node
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.grid_H = configs.grid_H
        self.grid_W = configs.grid_W
        self.grid_num = self.grid_H * self.grid_W
        self.n_neighbors = configs.n_neighbors
        self.feat_dim = configs.feat_dim

        self.register_buffer("distances", distances)
        self.register_buffer(
            "neighbors",
            torch.as_tensor(neighbors, dtype=torch.long),
            persistent=False,
        )
        self.time_emb = nn.Parameter(torch.empty(24, self.n_neighbors))
        nn.init.xavier_uniform_(self.time_emb)
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dim = F_hidden_dims[0]
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = 1

        self.F = nn.Sequential()
        self.F.add_module('conv1',
                          nn.Conv2d(in_channels=input_dim, out_channels=self.F_hidden_dim, kernel_size=self.kernel_size,
                                    stride=(1, 1), padding=self.padding))
        self.F.add_module('bn1', nn.GroupNorm(8, self.F_hidden_dim))
        self.conv_motion = nn.Conv2d(in_channels=self.feat_in, out_channels=2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.tanh = nn.Tanh()
        self.convgate = nn.Conv2d(in_channels=2,
                                  out_channels=1,
                                  kernel_size=(3, 3),
                                  padding=(1, 1), bias=self.bias)
        self.convgate1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()

        sample_tensor = torch.zeros(1, 1, self.grid_W, self.grid_H)
        self.register_buffer("grid0", make_grid(sample_tensor))
        sample_tensor = torch.zeros(1, 1, self.grid_W // 2, self.grid_H // 2)
        self.register_buffer("grid1", make_grid(sample_tensor))
        sample_tensor = torch.zeros(1, 1, self.grid_W // 4, self.grid_H // 4)
        self.register_buffer("grid2", make_grid(sample_tensor))

        self.weight = nn.Sequential(
            nn.Linear(3 * self.n_neighbors, 3 * self.n_neighbors),
            nn.ReLU(),
            nn.Linear(3 * self.n_neighbors, self.n_neighbors),
            nn.Sigmoid()
        )

        self.conv = nn.Conv1d(in_channels=self.grid_num, out_channels=self.num_node, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(in_features=self.feat_dim+1, out_features=2)
        self.relu = nn.ReLU()

        self.intensity = nn.Parameter(torch.empty(1, 1, self.grid_W, self.grid_H))
        nn.init.constant_(self.intensity, val=1)

        parameter_shape = (1, 1, self.grid_W, self.grid_H)
        if self.feat_in == 1:
            self.w_xt = nn.ParameterList([nn.Parameter(torch.empty(parameter_shape))])
            self.w_ht = nn.ParameterList([nn.Parameter(torch.empty(parameter_shape))])
            self.b = nn.ParameterList([nn.Parameter(torch.empty(parameter_shape))])
            nn.init.xavier_uniform_(self.w_xt[0])
            nn.init.xavier_uniform_(self.w_ht[0])
            nn.init.xavier_uniform_(self.b[0])
        else:
            self.w_xt = nn.ParameterList([
                nn.Parameter(torch.empty(parameter_shape)) for _ in range(self.feat_in)
            ])
            self.w_ht = nn.ParameterList([
                nn.Parameter(torch.empty(parameter_shape)) for _ in range(self.feat_in)
            ])
            self.b = nn.ParameterList([
                nn.Parameter(torch.empty(parameter_shape)) for _ in range(self.feat_in)
            ])
            for i in range(self.feat_in):
                nn.init.xavier_uniform_(self.w_xt[i])
                nn.init.xavier_uniform_(self.w_ht[i])
                nn.init.xavier_uniform_(self.b[i])
        self.layer_norm = nn.LayerNorm([self.feat_in, self.grid_W, self.grid_H])

    def init_x(self, x_t):
        v = x_t[:, self.neighbors]
        x_grid = torch.sum(self.distances * v[..., 0], dim=-1)
        return x_grid

    def warp(self, input, flow, grid, mode="bilinear", padding_mode="zeros"):
        B, C, H, W = input.size()
        vgrid = grid + flow

        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = torch.nn.functional.grid_sample(input, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True) + self.intensity
        return output

    def forward(self, xt, x_cor, x1_cor, ht_1):
        batch_size = xt.shape[0]
        if self.feat_in == 1:
            x_grid = self.init_x(xt[..., [0]]).unsqueeze(1)
            x_grid = x_grid.reshape(
                batch_size, 1, self.grid_W, self.grid_H
            )
            x_cor_var = self.tanh(
                self.w_xt[0] * x_cor[:, [0]]
                + self.w_ht[0] * ht_1[:, [0]]
                + self.b[0]
            )
            combined_conv = self.convgate(torch.cat([x_grid, x_cor_var], dim=1))
            K = torch.sigmoid(combined_conv)
            ht = x_grid + K * (x_cor_var - x_grid)
        else:
            ht = []
            for i in range(self.feat_in):
                x_grid = self.init_x(xt[..., [i]]).unsqueeze(1)
                x_grid = x_grid.reshape(
                    batch_size, 1, self.grid_W, self.grid_H
                )
                x_cor_var = self.tanh(
                    self.w_xt[i] * x_cor[:, [i]]
                    + self.w_ht[i] * ht_1[:, [i]]
                    + self.b[i]
                )
                combined_conv = self.convgate(
                    torch.cat([x_grid, x_cor_var], dim=1)
                )
                K = torch.sigmoid(combined_conv)
                ht.append(x_grid + K * (x_cor_var - x_grid))
            ht = torch.cat(ht, dim=1)
        ht = self.layer_norm(ht)

        motion = self.tanh(self.conv_motion(ht))
        delta_ht = self.warp(ht, motion, self.grid0, mode="bilinear", padding_mode="border") - ht
        res = x1_cor - ht
        if self.feat_in == 1:
            alpha = self.convgate1(res)
        else:
            alpha = torch.cat([
                self.convgate1(res[:, [i]]) for i in range(self.feat_in)
            ], dim=1)
        ht1 = ht + alpha * delta_ht
        # ht1 = ht

        # 后处理
        xt1 = ht1.reshape(batch_size, self.feat_in, -1).transpose(1, 2)
        xt1 = self.conv(xt1).unsqueeze(1)

        return xt1, ht1


class CDPNetModel(nn.Module):
    def __init__(self, configs):
        super(CDPNetModel, self).__init__()
        self.feat_in = configs.feat_in
        self.input_dim = configs.input_dim
        self.num_node = configs.num_node
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.grid_H = configs.grid_H
        self.grid_W = configs.grid_W
        self.grid_num = self.grid_H * self.grid_W
        self.n_neighbors = configs.n_neighbors
        self.feat_dim = configs.feat_dim
        self.root_path = configs.root_path

        self.national_pos = load_station_coordinates(
            self.root_path,
            getattr(configs, "station_coords_path", None),
            expected_nodes=self.num_node,
        )

        self.knn = sklearn.neighbors.NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.national_pos[..., [0, 1]])

        lat_min = getattr(configs, "grid_lat_min", None)
        lat_max = getattr(configs, "grid_lat_max", None)
        lon_min = getattr(configs, "grid_lon_min", None)
        lon_max = getattr(configs, "grid_lon_max", None)
        lat_min = self.national_pos[:, 0].min() if lat_min is None else lat_min
        lat_max = self.national_pos[:, 0].max() if lat_max is None else lat_max
        lon_min = self.national_pos[:, 1].min() if lon_min is None else lon_min
        lon_max = self.national_pos[:, 1].max() if lon_max is None else lon_max
        if lat_min >= lat_max or lon_min >= lon_max:
            raise ValueError(
                "CDPNet interpolation bounds require min < max; got "
                f"lat=({lat_min}, {lat_max}), lon=({lon_min}, {lon_max})."
            )

        olon = np.linspace(lon_min, lon_max, self.grid_W)
        olat = np.linspace(lat_min, lat_max, self.grid_H)
        self.olon, self.olat = np.meshgrid(olon, olat)
        distances, neighbors = self.knn.kneighbors(
            np.concatenate((self.olat.reshape(-1, 1), self.olon.reshape(-1, 1)), axis=1), return_distance=True)
        self.norm_dist(distances)
        distances = torch.tensor(distances, dtype=torch.float32)
        # distances1 = (distances1 - distances1.mean()) / distances1.std()

        self.distances = distances
        self.neighbors = neighbors

        self.national_pos[:, 0] = self.national_pos[:, 0] / 90
        self.national_pos[:, 1] = self.national_pos[:, 1] / 180
        mean = self.national_pos[:, 2].mean()
        std = self.national_pos[:, 2].std()
        self.national_pos[:, 2] = (self.national_pos[:, 2] - mean) / std
        self.register_buffer(
            "national_pos_tensor",
            torch.as_tensor(self.national_pos, dtype=torch.float32),
            persistent=False,
        )

        self.evolution = Evolution(configs, self.distances, self.neighbors, input_shape=(self.grid_W, self.grid_H), input_dim=self.feat_dim, F_hidden_dims=[self.feat_dim], n_layers=1, kernel_size=(7, 7))

        # self.time_series_emb_layer = nn.Conv2d(
        #     in_channels=(self.input_dim-self.feat_in+1)*self.seq_len, out_channels=self.feat_dim, kernel_size=(1, 1), bias=True)

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim*self.seq_len, out_channels=self.feat_dim, kernel_size=(1, 1), bias=True)


        self.conv1 = nn.Conv1d(in_channels=self.num_node, out_channels=self.grid_num, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.num_node, out_channels=self.grid_num, kernel_size=1)

        self.mlp = nn.Sequential(*[MultiLayerPerceptron(self.feat_dim, self.feat_dim) for _ in range(3)])

        if self.feat_in == 1:
            self.fc = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.feat_dim, out_channels=self.pred_len,
                    kernel_size=(1, 1), bias=True
                )
            ])
        else:
            self.fc = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.feat_dim, out_channels=self.pred_len,
                    kernel_size=(1, 1), bias=True
                ) for _ in range(self.feat_in)
            ])

        # self.fc = nn.Conv2d(in_channels=self.feat_dim, out_channels=self.pred_len, kernel_size=(1, 1), bias=True)
        # self.fc1 = nn.Conv2d(in_channels=self.feat_dim, out_channels=self.pred_len, kernel_size=(1, 1), bias=True)
        # self.fc2 = nn.Conv2d(in_channels=self.feat_dim, out_channels=self.pred_len, kernel_size=(1, 1), bias=True)


        self.drop = nn.Dropout(p=0.15)
        self.register_buffer("pad", torch.ones(size=[1, self.seq_len+self.pred_len+12, self.num_node, 2]))

    def norm_dist(self, distances):
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                distances[i][j] = 1 / distances[i][j]
            s = sum(distances[i])
            for j in range(distances.shape[1]):
                distances[i][j] = distances[i][j] / s

    def forward(self, x_enc, x_mark, is_training=True, epoch=-1):

        batch_size, _, num_nodes, _ = x_enc.shape

        x_mark_enc = x_mark.clone()
        x_mark_enc[..., 0] = x_mark_enc[..., 0] / 24 - 0.5
        x_mark_enc[..., 3] = x_mark_enc[..., 3] / 366 - 0.5

        national_pos = self.national_pos_tensor.to(dtype=x_enc.dtype).unsqueeze(0).unsqueeze(2)
        national_pos_repeat = national_pos.repeat(batch_size, 1, self.seq_len, 1)

        time_series = x_enc.transpose(1, 2).contiguous()
        time_series = torch.cat([time_series, x_mark_enc[..., [0, 3]].unsqueeze(1).repeat(1, self.num_node, 1, 1), national_pos_repeat], dim=-1)
        time_series_emb = time_series.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(time_series_emb)

        x_enc_deep_mlp = self.mlp(time_series_emb)
        if self.feat_in == 1:
            x_dec_deep = self.fc[0](x_enc_deep_mlp)
        else:
            x_dec_deep = torch.cat([
                self.fc[i](x_enc_deep_mlp) for i in range(self.feat_in)
            ], dim=-1)
        # x_dec_deep = torch.cat([self.fc1(x_enc_deep_mlp), self.fc2(x_enc_deep_mlp)], dim=-1)

        if epoch == 0:
            return x_dec_deep

        x_out = []

        if self.feat_in == 1:
            ht = x_enc[:, [0], :, 0]
            ht = self.conv1(ht.transpose(1, 2)).transpose(1, 2).view(
                batch_size, 1, self.grid_W, self.grid_H
            )
        else:
            ht = []
            for i in range(self.feat_in):
                ht_var = x_enc[:, [0], :, i]
                ht_var = self.conv1(ht_var.transpose(1, 2)).transpose(1, 2).view(
                    batch_size, 1, self.grid_W, self.grid_H
                )
                ht.append(ht_var)
            ht = torch.cat(ht, dim=1)

        x_enc_dec = torch.cat([x_enc, x_dec_deep], dim=1)

        for ei in range(1, self.seq_len + self.pred_len):
            if ei < self.seq_len:
                xt = x_enc_dec[:, ei - 1]
            else:
                xt = x_out[-1].squeeze(1)
            x_cor = x_enc_dec[:, ei - 1]
            x_cor = self.conv1(x_cor).transpose(1, 2).reshape(
                batch_size, self.feat_in, self.grid_W, self.grid_H
            )
            x1_cor = x_enc_dec[:, ei]
            x1_cor = self.conv1(x1_cor).transpose(1, 2).reshape(
                batch_size, self.feat_in, self.grid_W, self.grid_H
            )

            xt1, ht = self.evolution(xt, x_cor, x1_cor, ht)

            x_out.append(xt1)

        dec_out = torch.cat(x_out, dim=1)


        if is_training:
            return dec_out
        else:
            return dec_out[:, -self.pred_len:]


class Model(nn.Module):
    """TSLib tensor-layout adapter around the CDPNet network."""

    def __init__(self, configs):
        super().__init__()
        self.num_nodes = getattr(configs, "num_nodes", getattr(configs, "num_node", 1))
        self.num_features = 1
        configs.num_node = self.num_nodes
        configs.feat_in = self.num_features
        configs.input_dim = self.num_features + 2 + 3
        self.model = CDPNetModel(configs)
        self.training_epoch = None

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.ndim != 4:
            raise ValueError(f"CDPNet expects [B,L,N,V], got {tuple(x_enc.shape)}.")
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
                f"CDPNet was configured for {self.num_nodes} stations, got "
                f"{x_enc.shape[-1]}."
            )
        x_enc = x_enc.unsqueeze(-1)
        if x_mark_enc is None:
            raise ValueError("CDPNet requires calendar marks as x_mark_enc.")
        is_training = self.training_epoch is not None
        epoch = -1 if self.training_epoch is None else self.training_epoch
        output = self.model(
            x_enc, x_mark_enc, is_training=is_training, epoch=epoch
        )
        output = output.flatten(start_dim=2)
        return output.reshape(
            batch_size, num_variables, output.shape[1], num_stations
        ).permute(0, 2, 3, 1).contiguous()
