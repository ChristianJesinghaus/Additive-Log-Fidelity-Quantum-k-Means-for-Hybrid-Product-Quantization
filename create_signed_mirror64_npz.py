# -*- coding: utf-8 -*-
"""
Create a synthetic signed dataset for the sign-aware ablation.

We generate 5 prototype directions in R^64 and mirror them, yielding
10 classes:
    +mu_0, -mu_0, +mu_1, -mu_1, ..., +mu_4, -mu_4
"""
from pathlib import Path
import argparse
import numpy as np


def build_signed_mirror64(
    n_pairs: int = 5,
    samples_per_class: int = 300,
    dim: int = 64,
    noise_std: float = 0.12,
    seed: int = 13,
):
    rng = np.random.default_rng(seed)

    prototypes = rng.normal(size=(n_pairs, dim))
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-12

    X_parts = []
    y_parts = []

    for j in range(n_pairs):
        for sign_idx, sign in enumerate((+1.0, -1.0)):
            label = 2 * j + sign_idx
            center = sign * prototypes[j]
            samples = center + noise_std * rng.normal(size=(samples_per_class, dim))
            X_parts.append(samples.astype(np.float32))
            y_parts.append(np.full(samples_per_class, label, dtype=np.int64))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="datasets/signed_mirror64_full.npz",
        help="Output NPZ path",
    )
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--noise-std", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    X, y = build_signed_mirror64(
        samples_per_class=args.samples_per_class,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        data=X,
        labels=y,
        dataset_name="signed_mirror64_full",
        feature_shape=np.array([64], dtype=np.int64),
    )
    print(f"[INFO] Saved {len(X)} samples to {out_path}")


if __name__ == "__main__":
    main()