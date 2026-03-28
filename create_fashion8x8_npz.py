# -*- coding: utf-8 -*-
"""
Create a 64-D Fashion-MNIST dataset by center-cropping 28x28 -> 24x24
and average-pooling 3x3 blocks to 8x8.

Output format:
    data   : (N, 64)
    labels : (N,)
"""
from pathlib import Path
import argparse
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder


def downsample_to_8x8(X28: np.ndarray) -> np.ndarray:
    imgs = X28.reshape(-1, 28, 28).astype(np.float32)

    # center crop 24x24 so that 3x3 average pooling lands exactly on 8x8
    imgs = imgs[:, 2:26, 2:26]

    # average pooling: (24,24) -> (8,8)
    imgs = imgs.reshape(-1, 8, 3, 8, 3).mean(axis=(2, 4))
    return imgs.reshape(-1, 64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="datasets/fashion_mnist_8x8_full.npz",
        help="Output NPZ path",
    )
    args = parser.parse_args()

    print("[INFO] Downloading Fashion-MNIST from OpenML (cached by scikit-learn)...")
    ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False)

    X = ds.data.astype(np.float32)
    y = LabelEncoder().fit_transform(ds.target)

    X64 = downsample_to_8x8(X)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        data=X64,
        labels=y.astype(np.int64),
        dataset_name="fashion_mnist_8x8_full",
        original_shape=np.array([28, 28], dtype=np.int64),
        feature_shape=np.array([8, 8], dtype=np.int64),
    )
    print(f"[INFO] Saved {len(X64)} samples to {out_path}")


if __name__ == "__main__":
    main()