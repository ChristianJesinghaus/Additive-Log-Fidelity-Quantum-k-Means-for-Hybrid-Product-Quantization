# -*- coding: utf-8 -*-
"""
Create an unsplit full Digits dataset for the paper protocol.
Output format:
    data   : (N, D)
    labels : (N,)
"""
from pathlib import Path
import argparse
import numpy as np
from sklearn.datasets import load_digits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="datasets/digits64_full.npz",
        help="Output NPZ path",
    )
    args = parser.parse_args()

    ds = load_digits()
    X = ds.data.astype(np.float32)
    y = ds.target.astype(np.int64)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        data=X,
        labels=y,
        dataset_name="digits64_full",
        feature_shape=np.array([8, 8], dtype=np.int64),
    )
    print(f"[INFO] Saved {len(X)} samples to {out_path}")


if __name__ == "__main__":
    main()