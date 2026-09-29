import torch
import torch.nn as nn
import math
from layers.Corrformer_Embed import DataEmbedding
from layers.Corrformer_CausalConv import CausalConv
from layers.Corrformer_Correlation import AutoCorrelation, AutoCorrelationLayer, CrossCorrelation, CrossCorrelationLayer, \
    MultiCorrelation
from layers.Corrformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, \
    my_Layernorm, series_decomp


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        node_num_arg = getattr(configs, "node_num", None)
        canonical_node_num = getattr(configs, "num_nodes", None)
        explicit_counts = {
            value for value in (node_num_arg, canonical_node_num)
            if value not in (None, 1)
        }
        if len(explicit_counts) > 1:
            raise ValueError(
                "Corrformer received conflicting station counts: "
                f"node_num={node_num_arg}, num_nodes={canonical_node_num}."
            )
        configs.node_num = explicit_counts.pop() if explicit_counts else (
            node_num_arg or canonical_node_num or 1
        )
        configs.num_nodes = configs.node_num
        if configs.node_list == [1] and configs.node_num != 1:
            configs.node_list = [configs.node_num]
        if math.prod(configs.node_list) != configs.node_num:
            raise ValueError(
                f"Corrformer node_list={configs.node_list} must multiply to "
                f"node_num={configs.node_num}."
            )
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.node_num = configs.node_num
        # Full FP32 saved activations for Global exceed a 32-GiB GPU even at B=1.
        # Offload autograd's saved tensors only; parameters/compute stay on GPU.
        self.cpu_offload = getattr(configs, "corrformer_cpu_offload", self.node_num >= 2000)
        if self.cpu_offload:
            print("CORRFORMER_MEMORY saved_activations=cpu precision=fp32", flush=True)
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out
        self.node_list = configs.node_list  # node_num = node_list[0]*node_list[1]*node_list[2]...
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Encoding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.root_path,
                                           configs.node_num, configs.embed, configs.freq,
                                           configs.dropout, getattr(configs, "station_coords_path", None))
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.root_path,
                                           configs.node_num, configs.embed, configs.freq,
                                           configs.dropout, getattr(configs, "station_coords_path", None))

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MultiCorrelation(
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor_temporal, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        CrossCorrelationLayer(
                            CrossCorrelation(
                                CausalConv(
                                    num_inputs=configs.d_model // configs.n_heads * configs.seq_len,
                                    num_channels=[configs.d_model // configs.n_heads * configs.seq_len] \
                                                 * configs.dec_tcn_layers,
                                    kernel_size=3),
                                False, configs.factor_spatial, attention_dropout=configs.dropout,
                                output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        configs.node_num,
                        configs.node_list,
                        dropout=configs.dropout,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    MultiCorrelation(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor_temporal, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        CrossCorrelationLayer(
                            CrossCorrelation(
                                CausalConv(
                                    num_inputs=configs.d_model // configs.n_heads * (self.label_len + self.pred_len),
                                    num_channels=[configs.d_model // configs.n_heads * (self.label_len + self.pred_len)] \
                                                 * configs.dec_tcn_layers,
                                    kernel_size=3),
                                False, configs.factor_spatial, attention_dropout=configs.dropout,
                                output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        configs.node_num,
                        configs.node_list,
                        dropout=configs.dropout,
                    ),
                    MultiCorrelation(
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor_temporal, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        CrossCorrelationLayer(
                            CrossCorrelation(
                                CausalConv(
                                    num_inputs=configs.d_model // configs.n_heads * (self.label_len + self.pred_len),
                                    num_channels=[configs.d_model // configs.n_heads * (self.label_len + self.pred_len)] \
                                                 * configs.dec_tcn_layers,
                                    kernel_size=3),
                                False, configs.factor_spatial, attention_dropout=configs.dropout,
                                output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        configs.node_num,
                        configs.node_list,
                        dropout=configs.dropout,
                    ),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        if self.cpu_offload and self.training and torch.is_grad_enabled() and x_enc.is_cuda:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                return self._forward(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                     enc_self_mask, dec_self_mask, dec_enc_mask)
        return self._forward(x_enc, x_mark_enc, x_dec, x_mark_dec,
                             enc_self_mask, dec_self_mask, dec_enc_mask)

    def _forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        if x_enc.ndim != 4 or x_dec.ndim != 4:
            raise ValueError("Corrformer expects x_enc and x_dec as [B,L,N,V].")
        batch_size, _, num_stations, num_variables = x_enc.shape
        if x_dec.shape[0] != batch_size or x_dec.shape[2:] != (num_stations, num_variables):
            raise ValueError("Corrformer encoder and decoder layouts must match.")
        if num_stations != self.node_num:
            raise ValueError(
                f"Corrformer was configured for {self.node_num} stations but received "
                f"{num_stations}. Set --node_num or --num_nodes accordingly."
            )
        expected_channels = (self.enc_in, self.dec_in, self.c_out)
        if expected_channels != (num_variables,) * 3:
            raise ValueError(
                f"Corrformer received {num_variables} variables per station but "
                f"enc_in/dec_in/c_out={expected_channels}. Configure all three to "
                f"{num_variables} to match the source implementation."
            )

        x_enc = x_enc.reshape(batch_size, x_enc.shape[1], num_stations * num_variables)
        # init & normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = x_enc * self.affine_weight.repeat(1, 1, self.node_num) + self.affine_bias.repeat(1, 1, self.node_num)
        # decomp
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            [x_enc.shape[0], self.pred_len, x_enc.shape[2]],
            device=x_dec.device, dtype=x_dec.dtype,
        )
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input init
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        B, L, D = x_enc.shape
        _, _, C = x_mark_enc.shape
        x_enc = x_enc.view(B, L, self.node_num, -1).permute(0, 2, 1, 3).contiguous() \
            .view(B * self.node_num, L, D // self.node_num)
        x_mark_enc = x_mark_enc.unsqueeze(1).repeat(1, self.node_num, 1, 1).view(B * self.node_num, L, C)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        B, L, D = seasonal_init.shape
        _, _, C = x_mark_dec.shape
        seasonal_init = seasonal_init.view(B, L, self.node_num, -1).permute(0, 2, 1, 3).contiguous() \
            .view(B * self.node_num, L, D // self.node_num)
        trend_init = trend_init.view(B, L, self.node_num, -1).permute(0, 2, 1, 3).contiguous() \
            .view(B * self.node_num, L, D // self.node_num)
        x_mark_dec = x_mark_dec.unsqueeze(1).repeat(1, self.node_num, 1, 1).view(B * self.node_num, L, C)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        dec_out = dec_out[:, -self.pred_len:, :] \
            .view(B, self.node_num, self.pred_len, D // self.node_num).permute(0, 2, 1, 3).contiguous() \
            .view(B, self.pred_len, D)  # B L D

        # scale back
        dec_out = dec_out - self.affine_bias.repeat(1, 1, self.node_num)
        dec_out = dec_out / (self.affine_weight.repeat(1, 1, self.node_num) + 1e-10)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out.reshape(
            batch_size, self.pred_len, num_stations, num_variables
        )
