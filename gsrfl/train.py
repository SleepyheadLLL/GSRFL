"""Single GSRFL training run: seeding, data, server, and saving the round history and configuration."""

import argparse
import json
import os
import random
from dataclasses import fields

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from gsrfl.config import Config
from gsrfl.data import build_client_datasets
from gsrfl.server import FederatedServer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(cfg: Config):
    set_seed(cfg.run_seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    client_datasets, test_dataset, _ = build_client_datasets(cfg)
    cut = len(test_dataset) - min(max(cfg.heldout_size, 0), len(test_dataset) - 1)
    test_loader = DataLoader(Subset(test_dataset, range(cut)), batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    server = FederatedServer(cfg, client_datasets, test_loader, device)
    history = server.run()
    stem = os.path.join(cfg.save_dir, f"{cfg.experiment_name}_{cfg.run_tag}")
    with open(f"{stem}_history.json", "w") as f:
        json.dump(history, f, indent=1)
    with open(f"{stem}_config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=1)
    return history, server


def main() -> None:
    ap = argparse.ArgumentParser(description="Single GSRFL run")
    for f in fields(Config):
        default = getattr(Config, f.name)
        if isinstance(default, bool):
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=default)
        elif default is None:
            ap.add_argument(f"--{f.name}", type=int, default=None)
        else:
            ap.add_argument(f"--{f.name}", type=type(default), default=default)
    run(Config(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
