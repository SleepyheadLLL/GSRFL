# GSRFL

## Requirements

- Python 3.10
- A CUDA GPU is recommended. All scripts also run on CPU with `--device cpu`.
- Packages: `torch`, `torchvision`, `numpy`, and `matplotlib` (only for the heatmap).

```bash
pip install -r requirements.txt
```

The experiments were run with Python 3.10, torch 2.11, torchvision 0.26, numpy 1.26 and matplotlib 3.10.

## Data

FashionMNIST and CIFAR-10 are downloaded automatically into `./data`. Tiny-ImageNet must be prepared once:

```bash
python prepare_tinyimagenet.py --data_dir ./data
```

## Running the experiments

Run all commands from this directory. Each suite writes one CSV row per run to `results/<suite>/<suite>.csv`, a per-configuration summary (mean and standard error) to `results/<suite>/<suite>_summary.csv`, and the round-by-round history of every run to `results/<suite>/runs/`. A suite that is interrupted resumes from the runs already saved.

Main comparison, recovery diagnostics, and cost (W in {10, 20}, M in {10, 20}, three repeats):

```bash
python run_experiments.py --suite main
```

Ablation and regularization sweep on CIFAR-10 at M = 20:

```bash
python run_experiments.py --suite ablation
```

Client membership changes (join and replace; refresh, Nystrom, and off-graph; seeds 42, 43, 44):

```bash
python run_experiments.py --suite churn
```

Data heterogeneity (shard, Dirichlet 0.5, Dirichlet 0.1; seeds 42, 43, 44):

```bash
python run_experiments.py --suite partition
```

Client similarity heatmap on CIFAR-10:

```bash
python make_heatmap.py
```

Common options: `--datasets fashionmnist cifar10 tinyimagenet`, `--device cpu`, `--data_dir`, `--out`, `--seeds`, and `--repeats`.

## A single run

Every field of `gsrfl/config.py` is a command-line option. The defaults are the paper settings.

```bash
python -m gsrfl.train --dataset cifar10 --sample_clients 20 --low_freq_rank 8 --warmup_rounds 10
```

Accuracy is measured on the test split without its last 1000 images, which are held out (`--heldout_size`).
