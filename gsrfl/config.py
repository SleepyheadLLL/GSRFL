"""Run configuration for GSRFL. Defaults are the settings used in the paper."""

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class Config:
    seed: int = 42
    repeat: int = 0
    device: str = "cuda"

    dataset: str = "cifar10"
    data_dir: str = "./data"
    num_clients: int = 100
    samples_per_client: int = 0
    partition: str = "shard"
    shard_per_client: int = 0
    dirichlet_alpha: float = 0.5
    partition_seed: int = 0
    heldout_size: int = 1000

    rounds: int = 200
    warmup_rounds: int = 10
    local_epochs: int = 5
    batch_size: int = 64
    lr: float = 0.1
    momentum: float = 0.9
    server_lr: float = 1.0

    sample_clients: int = 20
    aggregation: str = "gsrfl"
    low_freq_rank: int = 8
    ridge_lambda: float = 1e-2
    recovery_ema: float = 0.7
    server_ema: float = 0.0

    churn_enabled: bool = False
    churn_strategy: str = "offgraph"
    churn_initial_active: Optional[int] = None
    churn_interval: int = 47
    churn_n_leave: int = 0
    churn_n_join: int = 10
    churn_min_neighbours: int = 10

    save_dir: str = "./results/runs"
    experiment_name: str = "gsrfl"

    @property
    def run_seed(self) -> int:
        return self.seed if self.repeat == 0 else self.seed * 1000 + self.repeat

    @property
    def run_tag(self) -> str:
        return f"seed{self.seed}_rep{self.repeat}"

    def resolve_samples_per_client(self) -> int:
        if self.samples_per_client > 0:
            return self.samples_per_client
        return 1000 if self.dataset == "tinyimagenet" else 500

    def resolve_shard_per_client(self) -> int:
        if self.shard_per_client > 0:
            return self.shard_per_client
        return 20 if self.dataset == "tinyimagenet" else 2

    def to_dict(self) -> dict:
        return asdict(self)
