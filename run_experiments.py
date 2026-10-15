"""Runs the GSRFL experiment suites of the paper (main, ablation, churn, partition) and writes per-run and summary CSV files."""

import argparse
import csv
import json
import os
from dataclasses import replace

import numpy as np

from gsrfl.config import Config
from gsrfl.train import run

DATASETS = ("fashionmnist", "cifar10", "tinyimagenet")
SEEDS = (42, 43, 44)
REPORTED = ("best_acc", "final_acc", "uplink_mb", "server_time_s")
DIAGNOSTICS = ("median_sigma_min", "p5_sigma_min", "median_v_tau_ratio", "rho_dist", "n_placed")


def rank_for(m: int) -> int:
    return 4 if m <= 10 else 8


def run_cached(cfg: Config):
    path = os.path.join(cfg.save_dir, f"{cfg.experiment_name}_{cfg.run_tag}_history.json")
    if os.path.exists(path):
        print(f"[skip] {path}")
        with open(path) as f:
            return json.load(f)
    return run(cfg)[0]


def metrics(history) -> dict:
    sparse = [h for h in history if h.get("mode") == "gsrfl"]
    sig = [h["sigma_min"] for h in sparse]
    builds = [h["rho_dist"] for h in history if "rho_dist" in h]
    return {
        "best_acc": max(h["best_test_acc"] for h in history),
        "final_acc": history[-1]["test_accuracy"],
        "uplink_mb": history[-1]["uplink_mb"],
        "server_time_s": float(np.sum([h["server_time"] for h in history])),
        "median_sigma_min": float(np.median(sig)) if sig else "",
        "p5_sigma_min": float(np.percentile(sig, 5)) if sig else "",
        "median_v_tau_ratio": float(np.median([h["v_tau_ratio"] for h in sparse])) if sparse else "",
        "rho_dist": builds[-1] if builds else "",
        "n_placed": int(sum(h.get("churn_placed", 0) for h in history)),
    }


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, ident):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in ident), []).append(r)
    out = []
    for key, members in groups.items():
        row = dict(zip(ident, key), n_runs=len(members))
        for k in REPORTED:
            v = np.array([m[k] for m in members], dtype=float)
            row[f"{k}_mean"] = float(v.mean())
            row[f"{k}_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        for k in DIAGNOSTICS:
            v = [m[k] for m in members if m[k] != ""]
            row[f"{k}_mean"] = float(np.mean(v)) if v else ""
        out.append(row)
    return out


def execute(cells, out, name, ident):
    rows = []
    for ident_vals, cfg in cells:
        rows.append({**ident_vals, **metrics(run_cached(cfg))})
        write_csv(os.path.join(out, f"{name}.csv"), rows)
        write_csv(os.path.join(out, f"{name}_summary.csv"), summarize(rows, ident))


def main_suite(base, out, datasets, repeats):
    cells = []
    for ds in datasets:
        for w in (10, 20):
            for m in (10, 20):
                for rep in range(repeats):
                    cfg = replace(base, dataset=ds, warmup_rounds=w, sample_clients=m, low_freq_rank=rank_for(m),
                                  seed=42, repeat=rep, experiment_name=f"main_{ds}_W{w}_M{m}")
                    cells.append((dict(dataset=ds, W=w, M=m, K=rank_for(m), repeat=rep), cfg))
    execute(cells, out, "main", ("dataset", "W", "M", "K"))


def ablation_suite(base, out, datasets):
    arms = [
        ("fedavg", dict(aggregation="fedavg", server_ema=0.0)),
        ("fedavg_aggregate_ema", dict(aggregation="fedavg", server_ema=0.7)),
        ("gsrfl_beta0", dict(recovery_ema=0.0)),
        ("gsrfl_lambda0.001", dict(ridge_lambda=1e-3)),
        ("gsrfl", dict(ridge_lambda=1e-2)),
        ("gsrfl_lambda0.1", dict(ridge_lambda=1e-1)),
    ]
    cells = []
    for ds in datasets:
        for arm, over in arms:
            cfg = replace(base, dataset=ds, warmup_rounds=10, sample_clients=20, low_freq_rank=8,
                          seed=42, repeat=0, experiment_name=f"ablation_{ds}_{arm}", **over)
            cells.append((dict(dataset=ds, arm=arm), cfg))
    execute(cells, out, "ablation", ("dataset", "arm"))


def churn_suite(base, out, datasets, seeds, n_churn=10, n_events=3):
    start = base.num_clients - n_churn * n_events
    interval = max((base.rounds - base.warmup_rounds) // (n_events + 1), 1)
    cells = []
    for ds in datasets:
        for m in (10, 20):
            for scen, n_leave in (("join", 0), ("replace", n_churn)):
                for arm in ("refresh", "nystrom", "offgraph"):
                    for seed in seeds:
                        cfg = replace(base, dataset=ds, sample_clients=m, low_freq_rank=rank_for(m), seed=seed,
                                      repeat=0, churn_enabled=True, churn_strategy=arm, churn_initial_active=start,
                                      churn_interval=interval, churn_n_leave=n_leave, churn_n_join=n_churn,
                                      experiment_name=f"churn_{ds}_{scen}_{arm}_M{m}")
                        cells.append((dict(dataset=ds, M=m, scenario=scen, arm=arm, seed=seed), cfg))
    execute(cells, out, "churn", ("dataset", "M", "scenario", "arm"))


def partition_suite(base, out, datasets, seeds):
    settings = (("shard", 0.5), ("dirichlet", 0.5), ("dirichlet", 0.1))
    cells = []
    for ds in datasets:
        for part, alpha in settings:
            for seed in seeds:
                cfg = replace(base, dataset=ds, partition=part, dirichlet_alpha=alpha, sample_clients=20,
                              low_freq_rank=8, warmup_rounds=10, seed=seed, repeat=0,
                              experiment_name=f"partition_{ds}_{part}{alpha if part == 'dirichlet' else ''}")
                label = f"dirichlet{alpha}" if part == "dirichlet" else "shard"
                cells.append((dict(dataset=ds, partition=label, seed=seed), cfg))
    execute(cells, out, "partition", ("dataset", "partition"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, choices=["main", "ablation", "churn", "partition"])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--out", default="./results")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--rounds", type=int, default=200)
    args = ap.parse_args()

    out = os.path.join(args.out, args.suite)
    base = Config(device=args.device, data_dir=args.data_dir, rounds=args.rounds, save_dir=os.path.join(out, "runs"))
    os.makedirs(base.save_dir, exist_ok=True)
    if args.suite == "main":
        main_suite(base, out, args.datasets or DATASETS, args.repeats)
    elif args.suite == "ablation":
        ablation_suite(base, out, args.datasets or ("cifar10",))
    elif args.suite == "churn":
        churn_suite(base, out, args.datasets or DATASETS, args.seeds)
    else:
        partition_suite(base, out, args.datasets or DATASETS, args.seeds)


if __name__ == "__main__":
    main()
