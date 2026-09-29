"""MIGN with a dense station-tensor to native DGL graph adapter."""

from pathlib import Path

import dgl
import torch
import torch.nn as nn

import importlib


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("MIGN requires PyYAML.") from exc

        config_path = getattr(configs, "mign_config", None)
        if config_path:
            with Path(config_path).open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            model_args, data_args = config["model_args"], config["data_args"]
        else:
            model_args = getattr(configs, "mign_model_args", None)
            data_args = getattr(configs, "mign_data_args", None)
        if model_args is None or data_args is None:
            raise ValueError("MIGN requires --mign_config or configs.mign_model_args and configs.mign_data_args.")
        network = importlib.import_module("layers.MIGN_Network")
        self.model_args = model_args
        self.data_args = data_args
        data_dir = getattr(configs, "mign_data_dir", None)
        if data_dir:
            self.data_args["data_dir"] = data_dir
        self.data_args.setdefault("refinement_level", 3)
        self.data_args.setdefault("neighbor", 10)
        self.input_length = data_args["input_length"]
        self.output_length = data_args["output_length"]
        self.feature = data_args.get("feature", getattr(configs, "mign_feature", "MXSPD"))
        self.data_args["feature"] = self.feature
        configs.mign_model_args = self.model_args
        configs.mign_data_args = self.data_args
        if self.feature is None:
            raise ValueError("MIGN data_args must define the target node type in feature.")
        self.model = network.HGNN_GCN_EDGE_WO_SH(model_args=model_args, data_args=data_args)
        template_path = getattr(configs, "mign_graph_template", None)
        if template_path is None:
            candidates = sorted(Path(self.data_args["data_dir"]).rglob("*.bin"))
            if not candidates:
                raise FileNotFoundError(
                    "MIGN needs a static DGL graph template. Pass "
                    "--mign_graph_template or place a generated .bin graph under data_dir."
                )
            template_path = candidates[0]
        graphs, self.sh_embedding = dgl.load_graphs(str(template_path))
        if len(graphs) != 1:
            raise ValueError(
                f"MIGN graph template must contain one graph, got {len(graphs)}."
            )
        self.graph_template = graphs[0]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.ndim != 4:
            raise ValueError(f"MIGN expects [B,L,N,V], got {tuple(x_enc.shape)}.")
        batch_size, length, num_stations, num_variables = x_enc.shape
        x_enc = x_enc.permute(0, 3, 1, 2).reshape(
            batch_size * num_variables, length, num_stations
        )
        graph_nodes = self.graph_template.num_nodes(self.feature)
        if x_enc.shape[-1] != graph_nodes:
            raise ValueError(
                f"MIGN graph template has {graph_nodes} {self.feature!r} nodes, "
                f"but the Dataset supplied {x_enc.shape[-1]} stations."
            )
        # DGL batches disjoint graph copies into one graph, allowing all batch
        # items and all four variables to share each GPU kernel launch.  MIGN's
        # no-SH configuration does not consume timestamps or graph labels.
        sample_count = x_enc.shape[0]
        graph = dgl.batch([self.graph_template] * sample_count).to(x_enc.device)
        for step in range(self.input_length):
            graph.nodes[self.feature].data[f"t{step}"] = (
                x_enc[:, step].reshape(sample_count * graph_nodes, 1)
            )
        output_graph = self.model(graph, None, None)
        prediction = torch.stack([
            output_graph.nodes[self.feature]
            .data[f"t{self.input_length + step}"]
            .reshape(sample_count, graph_nodes)
            for step in range(self.output_length)
        ], dim=1)
        return prediction.reshape(
            batch_size, num_variables, self.output_length, num_stations
        ).permute(0, 2, 3, 1).contiguous()
