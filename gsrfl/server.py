"""GSRFL server: full-participation warm-up, graph construction, sample-consistent ridge recovery with coefficient averaging, churn handling, and the FedAvg ablation arm."""

import time
from typing import Dict, List

import numpy as np
import torch

from gsrfl.client import ClientTrainer, assign_flat_delta, model_dim
from gsrfl.graph import build_graph_basis, ridge_recovery
from gsrfl.membership import MembershipSchedule, partition_recoverable
from gsrfl.models import build_model


def fedavg_direction(updates: Dict[int, np.ndarray], ids: List[int], p: Dict[int, float]) -> np.ndarray:
    ids = [int(i) for i in ids]
    Y = np.stack([updates[i] for i in ids], axis=0)
    w = np.array([p[i] for i in ids]) * (max(len(p), 1) / max(len(ids), 1))
    return w @ Y


def gsrfl_direction(updates, observed_ids, interpolated_rows, interpolated_ids, p, offgraph_ids=()) -> np.ndarray:
    obs_ids = [int(i) for i in observed_ids] + [int(i) for i in offgraph_ids]
    parts, weights = [], []
    if obs_ids:
        parts.append(np.stack([updates[i] for i in obs_ids], axis=0))
        weights.append(np.array([p[i] for i in obs_ids]))
    int_ids = [int(i) for i in interpolated_ids]
    if int_ids:
        parts.append(np.asarray(interpolated_rows, dtype=np.float64))
        weights.append(np.array([p[i] for i in int_ids]))
    Y = np.concatenate(parts, axis=0)
    w = np.concatenate(weights, axis=0)
    return (w / w.sum()) @ Y


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct / total


class FederatedServer:
    def __init__(self, config, client_datasets, test_loader, device):
        self.config = config
        self.client_datasets = client_datasets
        self.test_loader = test_loader
        self.device = device
        self.global_model = build_model(config.dataset).to(device)
        self.dim = model_dim(self.global_model)
        self.trainer = ClientTrainer(device, config.batch_size, config.local_epochs, config.lr, config.momentum)
        self.membership = MembershipSchedule(
            num_clients=config.num_clients, rounds=config.rounds, seed=config.run_seed,
            enabled=config.churn_enabled, initial_active=config.churn_initial_active,
            interval=config.churn_interval, n_leave=config.churn_n_leave, n_join=config.churn_n_join,
            protect_rounds=config.warmup_rounds)
        self.rng = np.random.default_rng(config.run_seed)
        self.basis = None
        self.pending_basis = None
        self.server_ema_state = None
        self.coeff_ema = None
        self.ema_active = False
        self.release_sum: Dict[int, np.ndarray] = {}
        self.release_count: Dict[int, int] = {}
        self.history: List[dict] = []

    def _refresh_at(self, r: int) -> bool:
        cfg = self.config
        return cfg.churn_enabled and cfg.churn_strategy == "refresh" and self.membership.changed_at(r)

    def _is_full_round(self, r: int) -> bool:
        return r <= self.config.warmup_rounds or self._refresh_at(r)

    def _train(self, ids: List[int]):
        updates, losses, steps = {}, [], []
        for cid in ids:
            updates[cid], loss, n = self.trainer.train(self.global_model, self.client_datasets[cid])
            losses.append(loss)
            steps.append(n)
        return updates, float(np.mean(losses)), int(np.mean(steps))

    def _apply(self, direction: np.ndarray) -> None:
        beta = self.config.server_ema
        if beta > 0.0 and self.ema_active:
            if self.server_ema_state is None:
                self.server_ema_state = direction.astype(np.float64, copy=True)
            else:
                self.server_ema_state *= beta
                self.server_ema_state += (1.0 - beta) * direction
            direction = self.server_ema_state
        assign_flat_delta(self.global_model, direction, -self.config.server_lr * self.config.lr)

    def _full_round(self, r: int, active: List[int], record: dict) -> None:
        updates, loss, steps = self._train(active)
        p = {c: 1.0 / len(active) for c in active}
        self._apply(fedavg_direction(updates, active, p))
        record.update(mode="full", num_selected=len(active), local_steps=steps, avg_local_loss=loss)
        for c in active:
            if c in self.release_sum:
                self.release_sum[c] += updates[c]
            else:
                self.release_sum[c] = updates[c].copy()
            self.release_count[c] = self.release_count.get(c, 0) + 1

        if self.config.aggregation == "gsrfl" and (r == self.config.warmup_rounds or self._refresh_at(r)):
            t0 = time.time()
            graph_input = np.stack([self.release_sum[c] / self.release_count[c] for c in active], axis=0)
            basis = build_graph_basis(graph_input, active, self.config.low_freq_rank)
            self.pending_basis = basis
            record["graph_build_time"] = time.time() - t0
            record["rho_dist"] = basis.subspace_residual(graph_input)
            record["dbar"] = basis.dbar
            self.release_sum.clear()
            self.release_count.clear()

    def _place_new_clients(self, active, sampled, updates, record) -> None:
        if self.basis is None:
            return
        on_graph = {int(v) for v in self.basis.vertices}
        gone = on_graph - set(int(a) for a in active)
        if gone:
            try:
                self.basis = self.basis.drop(sorted(gone))
            except ValueError:
                return
            on_graph = {int(v) for v in self.basis.vertices}
        joiners = [c for c in sampled if int(c) not in on_graph]
        nbrs = [c for c in sampled if int(c) in on_graph]
        if not joiners:
            return
        if len(nbrs) < max(self.config.churn_min_neighbours, 1):
            record["churn_placed"] = 0
            return
        try:
            self.basis = self.basis.extend(
                new_ids=joiners, new_updates=np.stack([updates[c] for c in joiners], axis=0),
                neighbour_ids=nbrs, neighbour_updates=np.stack([updates[c] for c in nbrs], axis=0))
        except ValueError:
            return
        self.coeff_ema = None
        record.update(churn_placed=len(joiners), churn_neighbours=len(nbrs))

    def _sparse_round(self, r: int, active: List[int], record: dict) -> None:
        cfg = self.config
        m = min(cfg.sample_clients, len(active))
        sampled = sorted(self.rng.choice(active, size=m, replace=False).tolist())
        updates, loss, steps = self._train(sampled)
        record.update(num_selected=len(sampled), local_steps=steps, avg_local_loss=loss)
        p_active = {c: 1.0 / len(active) for c in active}

        if cfg.aggregation == "fedavg":
            t0 = time.time()
            self._apply(fedavg_direction(updates, sampled, p_active))
            record.update(mode="fedavg", agg_time=time.time() - t0)
            return
        if cfg.aggregation != "gsrfl":
            raise ValueError(f"unknown aggregation: {cfg.aggregation}")

        if cfg.churn_enabled and cfg.churn_strategy == "nystrom":
            self._place_new_clients(active, sampled, updates, record)
        obs, obs_off, interp, unrep = partition_recoverable(active, self.basis.vertices, sampled)
        if not obs:
            self._apply(fedavg_direction(updates, sampled, p_active))
            record.update(mode="gsrfl_no_graph_rows")
            return

        t0 = time.time()
        obs_local = self.basis.local_index(obs)
        interp_local = self.basis.local_index(interp) if interp else np.array([], int)
        p_local = np.array([p_active.get(int(v), 0.0) for v in self.basis.vertices])
        res = ridge_recovery(self.basis, obs_local, np.stack([updates[c] for c in obs], axis=0),
                             p_local, interp_local, cfg.ridge_lambda)
        coeffs = res.coeffs
        beta = cfg.recovery_ema
        if beta > 0.0:
            if self.coeff_ema is None or self.coeff_ema.shape != coeffs.shape:
                self.coeff_ema = coeffs.astype(np.float64, copy=True)
            else:
                self.coeff_ema *= beta
                self.coeff_ema += (1.0 - beta) * coeffs
            coeffs = self.coeff_ema
        recovered = self.basis.Q[interp_local, :] @ coeffs
        self._apply(gsrfl_direction(updates, obs, recovered, interp, p_active, obs_off))
        record.update(mode="gsrfl", recovery_time=time.time() - t0, sigma_min=res.sigma_min,
                      v_tau_ratio=res.v_tau * max(len(obs), 1), n_observed=len(obs),
                      n_offgraph=len(obs_off), n_interpolated=len(interp), n_unrepresented=len(unrep))

    def run(self) -> List[dict]:
        best_acc, cum_uplink = 0.0, 0
        for r in range(1, self.config.rounds + 1):
            active = self.membership.active_at(r)
            record = {"round": r, "n_active": len(active)}
            full = self._is_full_round(r)
            self.ema_active = not full
            if full:
                self._full_round(r, active, record)
            else:
                self._sparse_round(r, active, record)
            if self.pending_basis is not None:
                self.basis = self.pending_basis
                self.pending_basis = None
                self.coeff_ema = None
            cum_uplink += record["num_selected"] * self.dim * 4
            record["uplink_mb"] = cum_uplink / 1024 / 1024
            record["test_accuracy"] = evaluate(self.global_model, self.test_loader, self.device)
            best_acc = max(best_acc, record["test_accuracy"])
            record["best_test_acc"] = best_acc
            record["server_time"] = (record.get("graph_build_time", 0.0) + record.get("recovery_time", 0.0)
                                     + record.get("agg_time", 0.0))
            self.history.append(record)
            msg = (f"[Round {r:03d}] mode={record['mode']:<6} sel={record['num_selected']:>3} "
                   f"loss={record['avg_local_loss']:.4f} acc={record['test_accuracy']:.4f} best={best_acc:.4f}")
            if "sigma_min" in record:
                msg += f" sigma_min={record['sigma_min']:.3f} Vtau/FA={record['v_tau_ratio']:.2f}"
            print(msg + f" up={record['uplink_mb']:.1f}MB", flush=True)
        return self.history
