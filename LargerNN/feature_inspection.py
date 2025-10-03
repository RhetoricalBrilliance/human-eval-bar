
"""
Feature inspection for TabResNetModel on a TRUE holdout set.
- Permutation Feature Importance (PFI)
- 1D/2D Partial Dependence Plots (PDP)
This script **refuses to run** on the full training dataset. You must provide:
  (A) --test_data  <parquet of the TEST split only>
      OR
  (B) --full_data <full parquet> AND --splits_dir <dir containing test_idx.npy>
Usage examples:
  A) python feature_inspection.py --test_data data/safe_test.parquet --ckpt evalbar_model_large.pt
  B) python feature_inspection.py --full_data data/normalized/normalized.parquet --splits_dir splits --ckpt evalbar_model_large.pt
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from dataloader import ParquetChessDataset, FEATURE_COLS
from model import TabResNetModel

def accuracy_from_logits(logits, y_true):
    preds = logits.argmax(dim=1)
    return (preds == y_true).float().mean().item()

def logloss_from_logits(logits, y_true, eps=1e-9):
    probs = F.softmax(logits, dim=1)
    ll = F.nll_loss(torch.log(probs.clamp(min=eps)), y_true, reduction='mean')
    return ll.item()

def load_model(input_dim, num_classes, ckpt_path, device):
    model = TabResNetModel(input_dim=input_dim, num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def permutation_importance(model, X, y, repeats=5, metric="accuracy", seed=42, device="cpu"):
    rng = np.random.default_rng(seed)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    with torch.no_grad():
        base_logits = model(X_t)
        if metric == "accuracy":
            base = accuracy_from_logits(base_logits, y_t)
        elif metric == "logloss":
            base = -logloss_from_logits(base_logits, y_t)  # negate for "higher is better"
        else:
            raise ValueError("metric must be 'accuracy' or 'logloss'")

    means, stds = [], []
    for j in range(X.shape[1]):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            Xt = torch.tensor(Xp, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = model(Xt)
                score = accuracy_from_logits(logits, y_t) if metric == "accuracy" else -logloss_from_logits(logits, y_t)
            drops.append(base - score)
        means.append(np.mean(drops))
        stds.append(np.std(drops))
    return np.array(means), np.array(stds), base

def plot_importance(means, stds, feature_names, title, outfile=None):
    order = np.argsort(means)[::-1]
    means, stds = means[order], stds[order]
    names = [feature_names[i] for i in order]

    plt.figure(figsize=(8,5))
    y = np.arange(len(names))
    plt.barh(y, means, xerr=stds)
    plt.yticks(y, names)
    plt.gca().invert_yaxis()
    plt.xlabel("Metric drop on permutation (higher = more important)")
    plt.title(title)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150)
    plt.show()

def partial_dependence_1d(model, X, feature_index, grid=None, target_class=0, device="cpu"):
    X_ref = X.copy()
    if grid is None:
        q = np.linspace(0.05, 0.95, 21)
        grid = np.quantile(X[:, feature_index], q)
        grid = np.unique(grid)
    pdp = []
    with torch.no_grad():
        for v in grid:
            X_sweep = X_ref.copy()
            X_sweep[:, feature_index] = v
            Xt = torch.tensor(X_sweep, dtype=torch.float32, device=device)
            logits = model(Xt)
            probs = F.softmax(logits, dim=1)[:, target_class].mean().item()
            pdp.append(probs)
    return grid, np.array(pdp)

def plot_pdp_1d(grid, pdp, feature_name, target_class, outfile=None):
    plt.figure(figsize=(6,4))
    plt.plot(grid, pdp, marker='o')
    plt.xlabel(f"{feature_name}")
    plt.ylabel(f"Avg P(class={target_class})")
    plt.title(f"PDP: {feature_name} → P(class={target_class})")
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150)
    plt.show()

def partial_dependence_2d(model, X, idx_a, idx_b, n_grid=30, target_class=0, device="cpu"):
    xa = np.linspace(np.quantile(X[:, idx_a], 0.05), np.quantile(X[:, idx_a], 0.95), n_grid)
    xb = np.linspace(np.quantile(X[:, idx_b], 0.05), np.quantile(X[:, idx_b], 0.95), n_grid)
    Z = np.zeros((n_grid, n_grid))
    with torch.no_grad():
        for i, va in enumerate(xa):
            X_a = X.copy()
            X_a[:, idx_a] = va
            for j, vb in enumerate(xb):
                X_ab = X_a.copy()
                X_ab[:, idx_b] = vb
                Xt = torch.tensor(X_ab, dtype=torch.float32, device=device)
                logits = model(Xt)
                Z[i, j] = F.softmax(logits, dim=1)[:, target_class].mean().item()
    return xa, xb, Z

def plot_pdp_2d(xa, xb, Z, name_a, name_b, target_class, outfile=None):
    plt.figure(figsize=(6,5))
    plt.imshow(Z, aspect='auto', origin='lower',
               extent=[xb[0], xb[-1], xa[0], xa[-1]])
    plt.colorbar(label=f"Avg P(class={target_class})")
    plt.xlabel(name_b)
    plt.ylabel(name_a)
    plt.title(f"2D PDP: ({name_a}, {name_b}) → P(class={target_class})")
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150)
    plt.show()

def load_holdout(args):
    """Return X_test, y_test using either a test parquet or split indices."""

    if args.test_data:
        # Case A: direct test parquet path (already split offline with train-only normalization)
        df = pd.read_parquet(args.test_data)
        X = df[FEATURE_COLS].astype('float32').to_numpy(copy=False)
        y = df["Result"].map({"1-0":0,"0-1":1,"1/2-1/2":2}).astype('int64').to_numpy(copy=False)
        return X, y

    # Case B: use full parquet plus a test index file
    assert args.full_data and args.splits_dir, "Provide --test_data OR (--full_data AND --splits_dir)"
    test_idx_path = os.path.join(args.splits_dir, "test_idx.npy")
    if not os.path.exists(test_idx_path):
        raise FileNotFoundError(f"Expected {test_idx_path} with test indices.")
    test_idx = np.load(test_idx_path)

    df = pd.read_parquet(args.full_data)
    X = df.loc[test_idx, FEATURE_COLS].astype('float32').to_numpy(copy=False)
    y = df.loc[test_idx, "Result"].map({"1-0":0,"0-1":1,"1/2-1/2":2}).astype('int64').to_numpy(copy=False)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    # Mutually exclusive: either test_data OR (full_data + splits_dir)
    ap.add_argument('--test_data', type=str, default=None, help='Parquet containing ONLY the test split (preferred).')
    ap.add_argument('--full_data', type=str, default=None, help='Full parquet (used with --splits_dir).')
    ap.add_argument('--splits_dir', type=str, default=None, help='Directory containing test_idx.npy (used with --full_data).')

    ap.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint (.pt).')
    ap.add_argument('--metric', type=str, default='accuracy', choices=['accuracy','logloss'])
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--target_class', type=int, default=0, help='Class for PDP probability.')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    # Enforce holdout-only evaluation
    if args.test_data is None and (args.full_data is None or args.splits_dir is None):
        raise SystemExit("Refusing to run: provide --test_data OR (--full_data AND --splits_dir).")

    # Load holdout set ONLY
    X, y = load_holdout(args)
    input_dim = X.shape[1]

    # Load model
    model = load_model(input_dim, num_classes=3, ckpt_path=args.ckpt, device=args.device)

    # Permutation importance
    means, stds, base = permutation_importance(model, X, y, repeats=args.repeats,
                                               metric=args.metric, device=args.device)
    title = f"Permutation Feature Importance (holdout {args.metric}={base:.4f})"
    plot_importance(means, stds, FEATURE_COLS, title, outfile='perm_importance.png')

    # PDP for top-3 features
    top_idx = np.argsort(means)[::-1][:3]
    for i in top_idx:
        grid, pdp = partial_dependence_1d(model, X, i, target_class=args.target_class, device=args.device)
        plot_pdp_1d(grid, pdp, FEATURE_COLS[i], args.target_class, outfile=f'pdp_{FEATURE_COLS[i]}.png')

    # 2D PDP for top-2 features
    if len(top_idx) >= 2:
        a, b = top_idx[0], top_idx[1]
        xa, xb, Z = partial_dependence_2d(model, X, a, b, target_class=args.target_class, device=args.device)
        plot_pdp_2d(xa, xb, Z, FEATURE_COLS[a], FEATURE_COLS[b], args.target_class, outfile='pdp2d_top2.png')

if __name__ == "__main__":
    main()
