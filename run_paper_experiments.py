# -*- coding: utf-8 -*-
from __future__ import annotations

import os

# ----------------------------------------------------------------------
# Windows / scikit-learn / loky robustness fix
#
# Reason:
# scikit-learn's KMeans may call _openmp_effective_n_threads().
# If OMP_NUM_THREADS is not set, that code path asks joblib/loky for the
# number of physical cores. On some Windows setups this triggers a WMIC /
# subprocess decoding failure. We avoid that path entirely and make the
# experiments reproducible by forcing 1 OpenMP thread.
# ----------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from product_quantization.normalize import normalize_data
from product_quantization.PQKNN import ProductQuantizationKNN
from product_quantization.quantum_pqknn import QuantumProductQuantizationKNN
from product_quantization.util import squared_euclidean_dist


PLANS: Dict[str, Dict[str, Any]] = {
    "smoke_digits": {
        "dataset_file": "digits64_full.npz",
        "test_size": 120,
        "train_sizes": [60],
        "seeds": [0],
        "normalize_data": True,
        "k": 9,
        "n": 8,
        "c": 10,
        "variants": [
            {"name": "exact_knn", "algorithm": "exact_knn"},
            {"name": "classical", "algorithm": "classical"},
            {
                "name": "quantum_exact",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "exact",
                "quantum_shots": 2000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
            {
                "name": "quantum_shot",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 1000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
        ],
    },
    "main_digits": {
        "dataset_file": "digits64_full.npz",
        "test_size": 300,
        "train_sizes": [60, 120, 200, 300],
        "seeds": [0, 1, 2],
        "normalize_data": True,
        "k": 9,
        "n": 8,
        "c": 10,
        "variants": [
            {"name": "exact_knn", "algorithm": "exact_knn"},
            {"name": "classical", "algorithm": "classical"},
            {
                "name": "quantum_exact",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "exact",
                "quantum_shots": 2000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
            {
                "name": "quantum_shot",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 2000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
        ],
    },


    "ablate_fashion_high_shots_size_variation": {
        "dataset_file": "fashion_mnist_8x8_full.npz",
        "test_size": 1000,
        "train_sizes": [100, 200, 300],
        "seeds": [0, 1, 2],
        "normalize_data": True,
        "k": 9,
        "n": 8,
        "c": 10,
        "variants": [
            {
                "name": "tau_1e-2_shots_7000_fashion",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 7000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
            {
                "name": "tau_5e-2_shots_7000_fashion",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 7000,
                "max_iter_qk": 100,
                "qk_tolerance": 5e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
        ],
    },


    "main_fashion": {
        "dataset_file": "fashion_mnist_8x8_full.npz",
        "test_size": 1000,
        "train_sizes": [100, 200, 300],
        "seeds": [0, 1, 2],
        "normalize_data": True,
        "k": 9,
        "n": 8,
        "c": 10,
        "variants": [
            {"name": "exact_knn", "algorithm": "exact_knn"},
            {"name": "classical", "algorithm": "classical"},
            {
                "name": "quantum_exact",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "exact",
                "quantum_shots": 2000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
            {
                "name": "quantum_shot",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 2000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
        ],
    },
    "ablate_tolerance_digits": {
        "dataset_file": "digits64_full.npz",
        "test_size": 300,
        "train_sizes": [250],
        "seeds": [0, 1, 2],
        "normalize_data": True,
        "k": 9,
        "n": 8,
        "c": 10,
        "variants": [
            {
                "name": "quantum_shot_tau_2e3",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 2000,
                "max_iter_qk": 100,
                "qk_tolerance": 2e-3,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
            {
                "name": "quantum_shot_tau_5e3",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 2000,
                "max_iter_qk": 100,
                "qk_tolerance": 5e-3,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
            {
                "name": "quantum_shot_tau_1e2",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 2000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
        ],
    },


        "ablate_tolerance_high_digits_size_variation": {
        "dataset_file": "digits64_full.npz",
        "test_size": 300,
        "train_sizes": [60, 120, 200, 300],
        "seeds": [0, 1, 2],
        "normalize_data": True,
        "k": 9, 
        "n": 8,
        "c": 10,
        "variants": [
            {
                "name": "tau_1e-2_shots_7000",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 7000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
            {
                "name": "tau_5e-2_shots_7000_size_variation",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 7000,
                "max_iter_qk": 100,
                "qk_tolerance": 5e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
        ],
    },


    


        "ablate_tolerance_high_digits": {
        "dataset_file": "digits64_full.npz",
        "test_size": 300,
        "train_sizes": [250],
        "seeds": [0],
        "normalize_data": True,
        "k": 9,
        "n": 8,
        "c": 10,
        "variants": [
            {
                "name": "tau_1e-2_shots_7000",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 7000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
            {
                "name": "tau_5e-2_shots_7000",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 7000,
                "max_iter_qk": 100,
                "qk_tolerance": 5e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
            {
                "name": "tau_1e-1_shots_7000",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 7000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-1,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
            {
                "name": "tau_1e-1_shots_5000",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 5000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-1,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
            {
                "name": "tau_5e-2_shots_5000",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 5000,
                "max_iter_qk": 100,
                "qk_tolerance": 5e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
        ],
    },


    "ablate_shots_digits": {
        "dataset_file": "digits64_full.npz",
        "test_size": 300,
        "train_sizes": [300],
        "seeds": [0, 1, 2],
        "normalize_data": True,
        "k": 9,
        "n": 8,
        "c": 10,
        "variants": [
            {
                "name": "quantum_exact",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "exact",
                "quantum_shots": 2000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
            {
                "name": "quantum_shot_1k",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 1000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
            {
                "name": "quantum_shot_2k",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 2000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
            {
                "name": "quantum_shot_5k",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "shot",
                "quantum_shots": 5000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
        ],
    },
    "ablate_metric_digits": {
        "dataset_file": "digits64_full.npz",
        "test_size": 300,
        "train_sizes": [300],
        "seeds": [0, 1, 2],
        "normalize_data": True,
        "k": 9,
        "n": 8,
        "c": 10,
        "variants": [
            {
                "name": "quantum_exact_logf",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "exact",
                "quantum_shots": 2000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
            {
                "name": "quantum_exact_omf",
                "algorithm": "quantum",
                "distance_metric": "one_minus_fidelity",
                "fidelity_mode": "exact",
                "quantum_shots": 2000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
        ],
    },
    "ablate_signaware_synth": {
        "dataset_file": "signed_mirror64_full.npz",
        "test_size": 500,
        "train_sizes": [100, 200],
        "seeds": [0, 1, 2, 3, 4],
        "normalize_data": True,
        "k": 9,
        "n": 8,
        "c": 10,
        "variants": [
            {"name": "exact_knn", "algorithm": "exact_knn"},
            {
                "name": "quantum_exact_raw",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "exact",
                "quantum_shots": 2000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": False,
            },
            {
                "name": "quantum_exact_signaware",
                "algorithm": "quantum",
                "distance_metric": "log_fidelity",
                "fidelity_mode": "exact",
                "quantum_shots": 2000,
                "max_iter_qk": 100,
                "qk_tolerance": 1e-2,
                "smooth_eps": 1e-3,
                "sign_aware_encoding": True,
            },
        ],
    },
}

PLAN_GROUPS = {
    "all_main": ["main_digits", "main_fashion"],
    "all_ablations": [
        "ablate_tolerance_digits",
        "ablate_shots_digits",
        "ablate_metric_digits",
        "ablate_signaware_synth",
    ],
    "all": [
        "main_digits",
        "main_fashion",
        "ablate_tolerance_digits",
        "ablate_shots_digits",
        "ablate_metric_digits",
        "ablate_signaware_synth",
    ],
}


def load_full_dataset(npz_path: Path):
    with np.load(npz_path, allow_pickle=True) as d:
        files = set(d.files)
        if {"data", "labels"}.issubset(files):
            X = d["data"]
            y = d["labels"]
        elif {"train_data", "train_labels", "test_data", "test_labels"}.issubset(files):
            X = np.concatenate([d["train_data"], d["test_data"]], axis=0)
            y = np.concatenate([d["train_labels"], d["test_labels"]], axis=0)
        else:
            raise ValueError(
                f"{npz_path} must contain either (data, labels) "
                f"or (train_data, train_labels, test_data, test_labels)"
            )

    X = np.asarray(X, dtype=np.float64)
    y = LabelEncoder().fit_transform(np.asarray(y))
    return X, y


def _balanced_class_quotas(y: np.ndarray, total: int) -> Dict[int, int]:
    classes, counts = np.unique(y, return_counts=True)
    raw = total * counts / counts.sum()
    base = np.floor(raw).astype(int)
    remainder = int(total - base.sum())
    order = np.argsort(-(raw - base))
    for idx in order[:remainder]:
        base[idx] += 1
    return {int(cls): int(q) for cls, q in zip(classes, base)}


def build_nested_train_subsets(y_pool: np.ndarray, train_sizes: List[int], seed: int) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y_pool)
    class_indices = {}
    for cls in classes:
        idx = np.where(y_pool == cls)[0]
        class_indices[int(cls)] = rng.permutation(idx)

    subsets = {}
    for M in sorted(train_sizes):
        quotas = _balanced_class_quotas(y_pool, M)
        chosen = []
        for cls in classes:
            cls = int(cls)
            q = quotas[cls]
            chosen.extend(class_indices[cls][:q].tolist())
        subsets[M] = np.array(sorted(chosen), dtype=np.int64)
    return subsets


def exact_distance_sums(train_x: np.ndarray, sample: np.ndarray) -> np.ndarray:
    diff = train_x - sample[None, :]
    return np.einsum("ij,ij->i", diff, diff)


def topk_indices_from_distance_sums(distance_sums: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k_eff = min(int(k), len(distance_sums))
    idx = np.argpartition(distance_sums, kth=k_eff - 1)[:k_eff]
    idx = idx[np.argsort(distance_sums[idx])]
    return idx, distance_sums[idx]


def majority_vote_with_distance(labels: np.ndarray, distances: np.ndarray) -> int:
    vals, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()
    tied = vals[counts == max_count]
    if len(tied) == 1:
        return int(tied[0])

    best_label = None
    best_sum = np.inf
    for label in tied:
        dsum = float(np.sum(distances[labels == label]))
        if dsum < best_sum:
            best_sum = dsum
            best_label = int(label)
    return int(best_label)


def exact_knn_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, k: int) -> np.ndarray:
    preds = []
    for sample in test_x:
        dist = exact_distance_sums(train_x, sample)
        idx, dsel = topk_indices_from_distance_sums(dist, k)
        preds.append(majority_vote_with_distance(train_y[idx], dsel))
    return np.asarray(preds, dtype=train_y.dtype)


def exact_topk(train_x: np.ndarray, sample: np.ndarray, k: int) -> np.ndarray:
    dist = exact_distance_sums(train_x, sample)
    idx, _ = topk_indices_from_distance_sums(dist, k)
    return idx


def classical_approx_distance_sums(model: ProductQuantizationKNN, sample: np.ndarray) -> np.ndarray:
    distances = np.empty((model.k, model.n), dtype=np.float64)
    for partition_idx in range(model.n):
        partition_start = partition_idx * model.partition_size
        if partition_idx == model.n - 1:
            part = sample[partition_start:]
        else:
            partition_end = (partition_idx + 1) * model.partition_size
            part = sample[partition_start:partition_end]
        centroids_partition = model.subvector_centroids[partition_idx]
        distances[:, partition_idx] = np.asarray(
            squared_euclidean_dist(part, centroids_partition), dtype=np.float64
        ).reshape(-1)

    distance_sums = np.zeros(len(model.compressed_data), dtype=np.float64)
    for partition_idx in range(model.n):
        distance_sums += distances[:, partition_idx][model.compressed_data[:, partition_idx]]
    return distance_sums


def quantum_approx_distance_sums(model: QuantumProductQuantizationKNN, sample: np.ndarray) -> np.ndarray:
    Dp = model._partition_dists(sample)
    distance_sums = np.zeros(len(model.compressed_data), dtype=np.float64)
    for partition_idx in range(model.n):
        distance_sums += Dp[:, partition_idx][model.compressed_data[:, partition_idx]]
    return distance_sums


def approximate_topk(model, sample: np.ndarray, k: int) -> np.ndarray:
    if isinstance(model, ProductQuantizationKNN):
        dist = classical_approx_distance_sums(model, sample)
    else:
        dist = quantum_approx_distance_sums(model, sample)
    idx, _ = topk_indices_from_distance_sums(dist, k)
    return idx


def retrieval_recall(train_x: np.ndarray, test_x: np.ndarray, model, k_eval: int) -> float:
    recalls = []
    for sample in test_x:
        gt = set(exact_topk(train_x, sample, k_eval).tolist())
        approx = set(approximate_topk(model, sample, k_eval).tolist())
        recalls.append(len(gt & approx) / float(k_eval))
    return float(np.mean(recalls))


def summarize_classical_training(model: ProductQuantizationKNN, train_size: int) -> Dict[str, Any]:
    hist = getattr(model, "kmeans_histories", {})
    if not hist:
        return {}

    n_iters = [float(m.get("n_iter", np.nan)) for m in hist.values()]
    inertias = [float(m.get("inertia", np.nan)) for m in hist.values()]
    return {
        "mean_partition_iters": float(np.nanmean(n_iters)),
        "max_partition_iters": float(np.nanmax(n_iters)),
        "loss_per_point": float(np.nanmean(inertias) / max(train_size, 1)),
    }


def summarize_quantum_training(model: QuantumProductQuantizationKNN, train_size: int) -> Dict[str, Any]:
    histories = getattr(model, "subvector_histories", {})
    if not histories:
        return {}

    lengths = []
    objectives = []
    accept_ratios = []
    backtracks = []
    rel_obj_changes = []
    shifts = []
    gauge_flips = []

    for hist in histories.values():
        if not hist:
            continue
        lengths.append(len(hist))
        last = hist[-1]
        objectives.append(float(last.get("objective_after_reassign", last.get("objective_after", np.nan))))
        accept_ratios.append(float(last.get("accept_ratio", np.nan)))
        backtracks.append(float(last.get("backtracks_mean", np.nan)))
        rel_obj_changes.append(float(last.get("relative_objective_change", np.nan)))
        shifts.append(float(last.get("shift", np.nan)))
        gauge_flips.append(float(last.get("gauge_flip_count", 0.0)))

    info = {}
    if lengths:
        info["mean_partition_iters"] = float(np.mean(lengths))
        info["max_partition_iters"] = float(np.max(lengths))
    if objectives:
        info["loss_per_point"] = float(np.mean(objectives) / max(train_size, 1))
    if accept_ratios:
        info["accept_ratio_last_mean"] = float(np.nanmean(accept_ratios))
    if backtracks:
        info["backtracks_last_mean"] = float(np.nanmean(backtracks))
    if rel_obj_changes:
        info["relative_objective_change_last_mean"] = float(np.nanmean(rel_obj_changes))
    if shifts:
        info["shift_last_mean"] = float(np.nanmean(shifts))
    if gauge_flips:
        info["gauge_flip_count_last_mean"] = float(np.nanmean(gauge_flips))

    try:
        qinfo = model.get_quantum_info()
        if isinstance(qinfo, dict):
            stats = qinfo.get("distance_stats", {})
            if isinstance(stats, dict):
                info["distance_exact_pairs"] = stats.get("exact_pairs", 0)
                info["distance_shot_pairs"] = stats.get("shot_pairs", 0)
                info["distance_fallback_pairs"] = stats.get("fallback_pairs", 0)
    except Exception:
        pass

    return info


def train_model(variant: Dict[str, Any], train_x: np.ndarray, train_y: np.ndarray, plan: Dict[str, Any], seed: int):
    algo = variant["algorithm"]
    if algo == "classical":
        model = ProductQuantizationKNN(
            n=plan["n"],
            c=0,
            k_clusters=plan["c"],
            random_state=seed,
            use_multiprocessing=False,
        )
    elif algo == "quantum":
        model = QuantumProductQuantizationKNN(
            n=plan["n"],
            c=plan["c"],
            max_iter_qk=variant.get("max_iter_qk", 100),
            quantum_shots=variant.get("quantum_shots", 2000),
            random_state=seed,
            distance_metric=variant.get("distance_metric", "log_fidelity"),
            smooth_eps=variant.get("smooth_eps", 1e-3),
            sign_aware_encoding=variant.get("sign_aware_encoding", False),
            fidelity_mode=variant.get("fidelity_mode", "shot"),
            qk_tolerance=variant.get("qk_tolerance", 1e-2),
        )
    else:
        raise ValueError(f"Unknown trainable algorithm: {algo}")

    t0 = time.perf_counter()
    model.compress(train_x, train_y)
    fit_time = float(time.perf_counter() - t0)
    return model, fit_time


def evaluate_trainable_model(model, train_x, train_y, test_x, test_y, k: int):
    t0 = time.perf_counter()
    preds = model.predict(test_x, k=k)
    predict_time = float(time.perf_counter() - t0)

    out = {
        "accuracy": float(accuracy_score(test_y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, preds)),
        "macro_f1": float(f1_score(test_y, preds, average="macro")),
        "predict_time_s": predict_time,
        "recall_at_1": float(retrieval_recall(train_x, test_x, model, 1)),
        "recall_at_10": float(retrieval_recall(train_x, test_x, model, 10)),
    }
    return out


def evaluate_exact_knn(train_x, train_y, test_x, test_y, k: int):
    t0 = time.perf_counter()
    preds = exact_knn_predict(train_x, train_y, test_x, k)
    predict_time = float(time.perf_counter() - t0)

    return {
        "accuracy": float(accuracy_score(test_y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, preds)),
        "macro_f1": float(f1_score(test_y, preds, average="macro")),
        "predict_time_s": predict_time,
        "recall_at_1": 1.0,
        "recall_at_10": 1.0,
    }


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def run_single_plan(plan_name: str, datasets_dir: Path, output_dir: Path, save_histories: bool = False) -> None:
    plan = PLANS[plan_name]
    dataset_path = datasets_dir / plan["dataset_file"]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    X_full, y_full = load_full_dataset(dataset_path)

    plan_dir = output_dir / plan_name
    plan_dir.mkdir(parents=True, exist_ok=True)
    save_json(plan_dir / "plan_spec.json", plan)

    all_rows = []

    for seed in plan["seeds"]:
        idx_all = np.arange(len(y_full))
        idx_pool, idx_test = train_test_split(
            idx_all,
            test_size=plan["test_size"],
            stratify=y_full,
            random_state=seed,
        )

        X_pool = X_full[idx_pool]
        y_pool = y_full[idx_pool]
        X_test = X_full[idx_test]
        y_test = y_full[idx_test]

        if plan.get("normalize_data", True):
            X_pool = normalize_data(X_pool)
            X_test = normalize_data(X_test)

        subset_indices = build_nested_train_subsets(y_pool, plan["train_sizes"], seed)

        split_payload = {
            "idx_pool": idx_pool,
            "idx_test": idx_test,
        }
        for M, idx_sub in subset_indices.items():
            split_payload[f"idx_train_{M}"] = idx_sub

        split_path = plan_dir / "splits" / f"{dataset_path.stem}_seed{seed}.npz"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(split_path, **split_payload)

        for M in sorted(plan["train_sizes"]):
            idx_sub = subset_indices[M]
            train_x = X_pool[idx_sub]
            train_y = y_pool[idx_sub]

            for variant in plan["variants"]:
                row = {
                    "plan": plan_name,
                    "dataset": dataset_path.stem,
                    "seed": int(seed),
                    "train_size": int(M),
                    "test_size": int(plan["test_size"]),
                    "variant_name": variant["name"],
                    "algorithm": variant["algorithm"],
                    "n": int(plan["n"]),
                    "c": int(plan["c"]),
                    "k_neighbors": int(plan["k"]),
                    "distance_metric": variant.get("distance_metric", "euclidean"),
                    "fidelity_mode": variant.get("fidelity_mode", "na"),
                    "quantum_shots": int(variant.get("quantum_shots", 0)),
                    "qk_tolerance": float(variant.get("qk_tolerance", np.nan))
                    if "qk_tolerance" in variant else np.nan,
                    "smooth_eps": float(variant.get("smooth_eps", np.nan))
                    if "smooth_eps" in variant else np.nan,
                    "sign_aware_encoding": bool(variant.get("sign_aware_encoding", False)),
                }

                print(
                    f"[RUN] plan={plan_name} dataset={dataset_path.stem} "
                    f"seed={seed} M={M} variant={variant['name']}"
                )

                if variant["algorithm"] == "exact_knn":
                    metrics = evaluate_exact_knn(train_x, train_y, X_test, y_test, plan["k"])
                    row["train_compress_time_s"] = 0.0
                else:
                    model, fit_time = train_model(variant, train_x, train_y, plan, seed)
                    metrics = evaluate_trainable_model(model, train_x, train_y, X_test, y_test, plan["k"])
                    row["train_compress_time_s"] = fit_time

                    if variant["algorithm"] == "classical":
                        row.update(summarize_classical_training(model, len(train_x)))
                    else:
                        row.update(summarize_quantum_training(model, len(train_x)))
                        if save_histories and hasattr(model, "export_histories"):
                            hist_dir = (
                                plan_dir
                                / "histories"
                                / f"{dataset_path.stem}_{variant['name']}_seed{seed}_M{M}"
                            )
                            model.export_histories(str(hist_dir))

                row.update(metrics)
                all_rows.append(row)

                row_path = (
                    plan_dir
                    / "runs"
                    / f"{dataset_path.stem}_{variant['name']}_seed{seed}_M{M}.json"
                )
                save_json(row_path, row)

    # write CSV summary
    csv_path = plan_dir / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({k for row in all_rows for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"[INFO] Summary written to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="smoke_digits",
        help=f"One of {list(PLANS.keys()) + list(PLAN_GROUPS.keys())}",
    )
    parser.add_argument("--datasets-dir", default="datasets")
    parser.add_argument("--output-dir", default="experiments/paper_runs")
    parser.add_argument("--save-histories", action="store_true")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plan in PLAN_GROUPS:
        plan_names = PLAN_GROUPS[args.plan]
    elif args.plan in PLANS:
        plan_names = [args.plan]
    else:
        raise ValueError(f"Unknown plan '{args.plan}'")

    for plan_name in plan_names:
        run_single_plan(
            plan_name=plan_name,
            datasets_dir=datasets_dir,
            output_dir=output_dir,
            save_histories=args.save_histories,
        )


if __name__ == "__main__":
    main()