"""Rebuilds the CIFAR-10 warm-up similarity matrix and plots the client similarity heatmap figure."""

import argparse
import json
import os
from collections import Counter

import numpy as np

from gsrfl.config import Config
from gsrfl.train import run


def plot(P, lead, stem, n_levels=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.5, "pdf.fonttype": 42})
    n = len(P)
    bounds = np.unique(np.quantile(P, np.linspace(0.0, 1.0, n_levels + 1)))
    cmap = LinearSegmentedColormap.from_list("coolwarm_soft", plt.get_cmap("coolwarm")(np.linspace(0.18, 0.82, 256)))
    norm = BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    edge = np.arange(n + 1)
    mesh = ax.pcolormesh(edge, edge, P, cmap=cmap, norm=norm, edgecolors="white", linewidth=0.12)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    for i in np.flatnonzero(lead[1:] != lead[:-1]) + 1:
        ax.axhline(i, color="k", lw=0.7, alpha=0.75)
        ax.axvline(i, color="k", lw=0.7, alpha=0.75)
    ticks = np.arange(0, n, 20)
    ax.set_xticks(ticks + 0.5)
    ax.set_yticks(ticks + 0.5)
    ax.set_xticklabels(ticks + 1)
    ax.set_yticklabels(ticks + 1)
    ax.tick_params(length=2)
    ax.set_xlabel("client (ordered by dominant label shard)")
    ax.set_ylabel("client")
    cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03, boundaries=bounds, ticks=bounds[::2])
    cb.ax.set_yticklabels([f"{b:.2f}" for b in bounds[::2]])
    cb.set_label(r"$P_{ij}$")
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=2)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="./results/heatmap")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = Config(dataset="cifar10", data_dir=args.data_dir, device=args.device, warmup_rounds=10, rounds=10,
                 sample_clients=20, low_freq_rank=8, seed=42, save_dir=os.path.join(args.out, "runs"),
                 experiment_name="heatmap_cifar10_W10")
    _, server = run(cfg)
    basis = server.basis
    labels = np.asarray(server.client_datasets[0].dataset.targets)
    keys = {}
    for cid, ds in server.client_datasets.items():
        top = [lab for lab, _ in Counter(int(labels[i]) for i in ds.indices).most_common(2)]
        keys[cid] = tuple(top + [-1] * (2 - len(top)))
    verts = [int(v) for v in basis.vertices]
    order = np.array(sorted(range(len(verts)), key=lambda i: keys[verts[i]]))
    P = np.asarray(basis.P)[np.ix_(order, order)]
    lead = np.array([keys[verts[i]][0] for i in order])

    iu = np.triu_indices(len(P), k=1)
    same = lead[iu[0]] == lead[iu[1]]
    summary = {"offdiag_mean": float(P[iu].mean()), "within_group_mean": float(P[iu][same].mean()),
               "across_group_mean": float(P[iu][~same].mean())}
    print(json.dumps(summary, indent=1))
    stem = os.path.join(args.out, "similarity_heatmap")
    with open(f"{stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    plot(P, lead, stem)


if __name__ == "__main__":
    main()
