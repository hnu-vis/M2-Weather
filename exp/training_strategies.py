"""Model-specific training semantics for the retained long-term baselines.

The experiment loop remains Time-Series-Library shaped, while this module
owns the differences that are part of each upstream implementation: loss,
optimizer, scheduler, clipping, auxiliary objectives and batch metadata.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


INFERENCE_ONLY_MODELS = {"Moirai", "Sundial", "TimeMoE", "Timer"}


def masked_mae(prediction, target, null_val=float("nan")):
    """BasicTS/HiSTGNN-compatible masked MAE."""
    if math.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        mask = target != null_val
    mask = mask.float()
    mask = mask / (mask.mean() + 1e-8)
    loss = torch.abs(prediction - target) * mask
    return torch.nan_to_num(loss).mean()


@dataclass(frozen=True)
class TrainingStrategy:
    name: str
    loss: str = "mse"
    optimizer: str = "adam"
    scheduler: str = "tslib"
    weight_decay: float = 0.0
    clip_grad: float | None = None
    auxiliary_loss: bool = False
    early_stopping_start_epoch: int = 0
    stop_on_patience: bool = True
    requires_upstream_batch: bool = False

    def make_optimizer(self, model, args):
        lr = getattr(args, "learning_rate", 1e-4)
        if self.name == "MIGN":
            unwrapped = model.module if isinstance(model, nn.DataParallel) else model
            lr = unwrapped.model_args.get("learning_rate", lr)
        override = getattr(args, "weight_decay", None)
        weight_decay = self.weight_decay if override is None else override
        cls = torch.optim.AdamW if self.optimizer == "adamw" else torch.optim.Adam
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError(f"{self.name} has no trainable parameters.")
        return cls(trainable_parameters, lr=lr, weight_decay=weight_decay)

    def make_criterion(self):
        if self.loss == "huber":
            return nn.HuberLoss(delta=0.5)
        if self.loss == "l1":
            return nn.L1Loss()
        if self.loss == "masked_mae":
            return masked_mae
        return nn.MSELoss()

    def make_scheduler(self, optimizer, args, steps_per_epoch):
        if self.scheduler == "one_cycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                steps_per_epoch=steps_per_epoch,
                pct_start=getattr(args, "pct_start", 0.3),
                epochs=getattr(args, "train_epochs", 10),
                max_lr=getattr(args, "learning_rate", 1e-4),
            )
        if self.scheduler == "multistep":
            milestones = getattr(args, "lr_milestones", [50])
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=getattr(args, "lr_gamma", 0.5)
            )
        if self.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=getattr(args, "t_max", 500),
                eta_min=optimizer.param_groups[0]["lr"] / 10,
            )
        if self.scheduler == "timerxl":
            if getattr(args, "cosine", True):
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=getattr(args, "tmax", None) or args.train_epochs,
                    eta_min=1e-8,
                )
            return None
        return None

    def batch_scheduler_step(self, scheduler, args):
        if self.scheduler == "one_cycle" and getattr(args, "lradj", "type1") == "TST":
            scheduler.step()

    def epoch_scheduler_step(self, scheduler, optimizer, args, epoch):
        if self.scheduler == "one_cycle":
            if getattr(args, "lradj", "type1") != "TST":
                from utils.tools import adjust_learning_rate
                adjust_learning_rate(optimizer, epoch, args)
        elif self.scheduler in {"multistep", "cosine"}:
            scheduler.step()
        elif self.scheduler == "timerxl":
            if scheduler is not None:
                scheduler.step()
            else:
                from utils.tools import adjust_learning_rate
                adjust_learning_rate(optimizer, epoch, args)
        elif self.scheduler == "duet":
            duet_schedule_epoch(optimizer, args, epoch)
        else:
            from utils.tools import adjust_learning_rate
            adjust_learning_rate(optimizer, epoch, args)

    def add_auxiliary_loss(self, loss, model, args):
        if not self.auxiliary_loss:
            return loss
        unwrapped = model.module if isinstance(model, nn.DataParallel) else model
        auxiliary = getattr(unwrapped, "auxiliary_loss", None)
        if auxiliary is not None:
            return loss + getattr(args, "aux_loss_weight", 1.0) * auxiliary
        return loss

    def clip(self, model, args):
        override = getattr(args, "clip_grad", None)
        value = self.clip_grad if override is None else override
        if value is not None and value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), value)


def duet_schedule_epoch(optimizer, args, epoch):
    """DUET upstream type3 schedule (with its model-specific default)."""
    base = args.learning_rate
    kind = getattr(args, "lradj", "type3")
    if kind == "type1":
        lr = base * (0.5 ** (epoch - 1))
    elif kind == "type2":
        values = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
        if epoch not in values:
            return
        lr = values[epoch]
    elif kind == "type3":
        lr = base if epoch < 3 else base * (0.9 ** (epoch - 3))
    elif kind == "constant":
        lr = base
    else:
        return
    for group in optimizer.param_groups:
        group["lr"] = lr
    print(f"Updating learning rate to {lr}")


_GENERIC = TrainingStrategy("TSLib")
STRATEGIES = {
    "Autoformer": TrainingStrategy("Autoformer"),
    "DLinear": TrainingStrategy("DLinear"),
    "iTransformer": TrainingStrategy("iTransformer"),
    "Corrformer": TrainingStrategy("Corrformer"),
    "PatchTST": TrainingStrategy("PatchTST", scheduler="one_cycle"),
    "TQNet": TrainingStrategy("TQNet", scheduler="one_cycle"),
    "DUET": TrainingStrategy("DUET", loss="huber", scheduler="duet", auxiliary_loss=True),
    "CDPNet": TrainingStrategy(
        "CDPNet", loss="l1", weight_decay=1e-4,
        early_stopping_start_epoch=1,
    ),
    "TimerXL": TrainingStrategy("TimerXL", scheduler="timerxl"),
    "STELLA": TrainingStrategy(
        "STELLA", loss="masked_mae", scheduler="multistep",
        weight_decay=5e-4, clip_grad=5.0,
    ),
    "HiSTGNN": TrainingStrategy(
        "HiSTGNN", loss="masked_mae", weight_decay=1e-4, clip_grad=5.0,
    ),
    "xPatch": TrainingStrategy("xPatch"),
    "S2Transformer": TrainingStrategy(
        "S2Transformer", weight_decay=1e-4, clip_grad=5.0,
    ),
    "EasyST": TrainingStrategy(
        "EasyST", weight_decay=0.0, clip_grad=5.0,
    ),
    "MIGN": TrainingStrategy(
        "MIGN", optimizer="adamw", scheduler="cosine",
        requires_upstream_batch=True,
    ),
}


def get_training_strategy(model_name):
    if model_name in INFERENCE_ONLY_MODELS:
        return TrainingStrategy(model_name)
    return STRATEGIES.get(model_name, _GENERIC)
