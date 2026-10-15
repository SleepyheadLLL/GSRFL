"""Dataset loading and client partitions (label shards and label Dirichlet)."""

import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms

_NORM = {
    "fashionmnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "tinyimagenet": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
}


def _transform(name: str):
    steps = [transforms.Resize((64, 64))] if name == "tinyimagenet" else []
    mean, std = _NORM[name]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def load_datasets(name: str, data_dir: str):
    tf = _transform(name)
    if name == "fashionmnist":
        train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tf)
        test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
    elif name == "cifar10":
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    elif name == "tinyimagenet":
        root = os.path.join(data_dir, "tiny-imagenet-200")
        if not os.path.isdir(os.path.join(root, "val_sorted")):
            raise FileNotFoundError(f"Tiny-ImageNet not prepared under {root}; run prepare_tinyimagenet.py")
        train = datasets.ImageFolder(os.path.join(root, "train"), transform=tf)
        test = datasets.ImageFolder(os.path.join(root, "val_sorted"), transform=tf)
    else:
        raise ValueError(f"unsupported dataset: {name}")
    targets = getattr(train, "targets", None)
    labels = np.asarray(targets if targets is not None else [y for _, y in train.samples])
    return train, test, labels


def shard_partition(labels: np.ndarray, num_clients: int, samples_per_client: int,
                    shard_per_client: int, seed: int) -> Dict[int, List[int]]:
    rng = np.random.default_rng(seed)
    total = num_clients * samples_per_client
    num_shards = num_clients * shard_per_client
    shard_size = total // num_shards
    if total > len(labels) or shard_size * num_shards != total:
        raise ValueError("invalid shard partition size")
    classes = np.unique(labels)
    by_class = {c: rng.permutation(np.flatnonzero(labels == c)) for c in classes}
    quota, remainder = divmod(total, len(classes))
    picked = [by_class[c][:quota + (1 if k < remainder else 0)] for k, c in enumerate(classes)]
    subset = np.concatenate(picked)
    ordered = subset[np.argsort(labels[subset], kind="stable")]
    shards = [ordered[i * shard_size:(i + 1) * shard_size].tolist() for i in range(num_shards)]
    rng.shuffle(shards)
    clients = defaultdict(list)
    ptr = 0
    for cid in range(num_clients):
        for _ in range(shard_per_client):
            clients[cid].extend(shards[ptr])
            ptr += 1
    return dict(clients)


def dirichlet_partition(labels: np.ndarray, num_clients: int, samples_per_client: int,
                        alpha: float, seed: int) -> Dict[int, List[int]]:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    by_class = {c: rng.permutation(np.flatnonzero(labels == c)).tolist() for c in classes}
    ptr = {c: 0 for c in classes}
    clients: Dict[int, List[int]] = {}
    for cid in range(num_clients):
        props = rng.dirichlet(np.full(len(classes), alpha))
        counts = np.floor(props * samples_per_client).astype(int)
        counts[-1] += samples_per_client - counts.sum()
        idx: List[int] = []
        for c, k in zip(classes, counts):
            pool = by_class[c]
            for _ in range(int(k)):
                if ptr[c] >= len(pool):
                    ptr[c] = 0
                idx.append(pool[ptr[c]])
                ptr[c] += 1
        clients[cid] = idx
    return clients


def build_client_datasets(cfg):
    train, test, labels = load_datasets(cfg.dataset, cfg.data_dir)
    n = cfg.resolve_samples_per_client()
    if cfg.partition == "shard":
        parts = shard_partition(labels, cfg.num_clients, n, cfg.resolve_shard_per_client(), cfg.partition_seed)
    elif cfg.partition == "dirichlet":
        parts = dirichlet_partition(labels, cfg.num_clients, n, cfg.dirichlet_alpha, cfg.partition_seed)
    else:
        raise ValueError(f"unsupported partition: {cfg.partition}")
    return {cid: Subset(train, idx) for cid, idx in parts.items()}, test, labels
