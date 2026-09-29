from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from data_provider.data_loader import (
    Dataset_French,
    Dataset_M3Europe,
    Dataset_M3France,
    Dataset_M3Global,
)


data_dict = {
    "French": Dataset_French,
    "M3France": Dataset_M3France,
    "M3Europe": Dataset_M3Europe,
    "M3Global": Dataset_M3Global,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag.lower(),
        size=[args.seq_len, args.label_len, args.pred_len],
        freq=args.freq,
    )
    train_stride = args.train_stride if flag.lower() == "train" else 1
    loader_data = (
        Subset(data_set, range(0, len(data_set), train_stride))
        if train_stride > 1 else data_set
    )
    distributed = bool(getattr(args, "use_ddp", False))
    is_main = bool(getattr(args, "is_main_process", True))
    if train_stride > 1:
        if is_main:
            print(
                f"{flag} {len(data_set)} sampled={len(loader_data)} "
                f"stride={train_stride}"
            )
    elif is_main:
        print(flag, len(data_set))
    batch_size = (
        args.batch_size if flag.lower() == "train"
        else (getattr(args, "eval_batch_size", None) or args.batch_size)
    )
    if flag.lower() != "train" and batch_size != args.batch_size and is_main:
        print(f"{flag} batch_size={batch_size} (train batch_size={args.batch_size})")

    sampler = None
    if distributed and flag.lower() == "train":
        sampler = DistributedSampler(
            loader_data, num_replicas=args.world_size, rank=args.rank,
            shuffle=True, seed=args.seed, drop_last=False,
        )
    elif distributed:
        # Do not pad validation with duplicates: the reduced metric stays exact.
        loader_data = Subset(
            loader_data, range(args.rank, len(loader_data), args.world_size)
        )

    data_loader = DataLoader(
        loader_data,
        batch_size=batch_size,
        shuffle=flag.lower() == "train" and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    return data_set, data_loader
