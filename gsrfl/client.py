"""Local client training with momentum SGD and per-epoch reshuffling; returns the update divided by the local learning rate."""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def flatten_params(model: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()]).astype(np.float64)


def model_dim(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def assign_flat_delta(model: nn.Module, flat: np.ndarray, scale: float) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            chunk = torch.from_numpy(np.ascontiguousarray(flat[offset:offset + n])).view_as(p).to(device=p.device, dtype=p.dtype)
            p.add_(scale * chunk)
            offset += n


class ClientTrainer:
    def __init__(self, device, batch_size: int, local_epochs: int, lr: float, momentum: float):
        self.device = device
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def train(self, global_model: nn.Module, dataset):
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0.0)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False)
        total_loss, total_n, steps = 0.0, 0, 0
        for _ in range(self.local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = self.loss_fn(model(x), y).mean()
                loss.backward()
                opt.step()
                total_loss += loss.item() * y.size(0)
                total_n += y.size(0)
                steps += 1
        update = (flatten_params(global_model) - flatten_params(model)) / self.lr
        return update, total_loss / max(total_n, 1), steps
