# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KNOWN_PLANS = [
    "main_digits",
    "main_fashion",
    "ablate_tolerance_digits",
    "ablate_tolerance_high_digits",
    "ablate_tolerance_high_digits_size_variation",
    "ablate_fashion_high_shots_size_variation",
    "ablate_shots_digits",
    "ablate_metric_digits",
    "ablate_signaware_synth",
]

MAIN_VARIANT_ORDER = ["exact_knn", "classical", "quantum_exact", "quantum_shot"]


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    out = df.copy()
    out.columns = [
        "_".join([str(x) for x in tup if str(x) != ""]).rstrip("_")
        for tup in out.columns.to_flat_index()
    ]
    return out


def ensure_dirs(base: Path) -> tuple[Path, Path, Path]:
    plots = base / "plots"
    tables = base / "tables"
    text = base / "text"
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    text.mkdir(parents=True, exist_ok=True)
    return plots, tables, text


def load_summary(base: Path, plan: str) -> pd.DataFrame | None:
    path = base / plan / "summary.csv"
    if not path.exists():
        print(f"[WARN] Missing summary for plan '{plan}': {path}")
        return None
    df = pd.read_csv(path)
    df["plan"] = plan
    return df


def aggregate_mean_std(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    group_cols = [c for c in group_cols if c in df.columns]
    metric_cols = [c for c in metric_cols if c in df.columns]
    agg = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    return flatten_columns(agg)


def ordered_variants_present(df: pd.DataFrame, order: list[str]) -> list[str]:
    present = list(df["variant_name"].dropna().unique())
    ordered = [v for v in order if v in present]
    ordered.extend([v for v in present if v not in ordered])
    return ordered


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(df.to_markdown(index=False), encoding="utf-8")
    except Exception:
        path.write_text(df.to_string(index=False), encoding="utf-8")


def save_latex_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(df.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
    except Exception:
        pass


def _get_yerr(sub: pd.DataFrame, std_col: str):
    if std_col not in sub.columns:
        return None
    return sub[std_col].fillna(0.0)


def plot_line_over_train_size(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in df.columns or "train_size" not in df.columns or "variant_name" not in df.columns:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for variant in ordered_variants_present(df, MAIN_VARIANT_ORDER):
        sub = df[df["variant_name"] == variant].sort_values("train_size")
        plt.errorbar(
            sub["train_size"],
            sub[mean_col],
            yerr=_get_yerr(sub, std_col),
            marker="o",
            capsize=3,
            label=str(variant),
        )
    plt.xlabel("Train size M")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bar(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in df.columns or "variant_name" not in df.columns:
        return

    sub = df.sort_values("variant_name")
    x = np.arange(len(sub))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(x, sub[mean_col], yerr=_get_yerr(sub, std_col), capsize=4)
    plt.xticks(x, sub["variant_name"], rotation=25, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_tolerance_lines(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in df.columns or "qk_tolerance" not in df.columns or "quantum_shots" not in df.columns:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for shots in sorted([s for s in df["quantum_shots"].dropna().unique()]):
        sub = df[df["quantum_shots"] == shots].sort_values("qk_tolerance")
        plt.errorbar(
            sub["qk_tolerance"],
            sub[mean_col],
            yerr=_get_yerr(sub, std_col),
            marker="o",
            capsize=3,
            label=f"shots={int(shots)}",
        )
    plt.xscale("log")
    plt.xlabel("qk_tolerance")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_shots_lines(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in df.columns:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))

    exact = df[df["variant_name"] == "quantum_exact"]
    shot = df[df["variant_name"] != "quantum_exact"].sort_values("quantum_shots")

    if not exact.empty:
        y = float(exact.iloc[0][mean_col])
        plt.axhline(y=y, linestyle="--", label="quantum_exact")

    if not shot.empty and "quantum_shots" in shot.columns:
        plt.errorbar(
            shot["quantum_shots"],
            shot[mean_col],
            yerr=_get_yerr(shot, std_col),
            marker="o",
            capsize=3,
            label="quantum_shot",
        )
    plt.xlabel("Shots")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_tolerance_size_variation(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in df.columns or "train_size" not in df.columns or "variant_name" not in df.columns:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for variant, sub in df.groupby("variant_name"):
        sub = sub.sort_values("train_size")
        plt.errorbar(
            sub["train_size"],
            sub[mean_col],
            yerr=_get_yerr(sub, std_col),
            marker="o",
            capsize=3,
            label=str(variant),
        )
    plt.xlabel("Train size M")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def recommend_tolerance(df: pd.DataFrame) -> dict:
    work = df.copy()
    if "accuracy_mean" not in work.columns:
        return {}

    best_acc = float(work["accuracy_mean"].max())
    best_rec = float(work["recall_at_10_mean"].max()) if "recall_at_10_mean" in work.columns else -np.inf

    cand = work[work["accuracy_mean"] >= best_acc - 0.01].copy()
    if "recall_at_10_mean" in cand.columns:
        cand = cand[cand["recall_at_10_mean"] >= best_rec - 0.01]
    if cand.empty:
        cand = work.copy()

    sort_cols = []
    asc = []
    if "mean_partition_iters_mean" in cand.columns:
        sort_cols.append("mean_partition_iters_mean")
        asc.append(True)
    if "train_compress_time_s_mean" in cand.columns:
        sort_cols.append("train_compress_time_s_mean")
        asc.append(True)
    if "qk_tolerance" in cand.columns:
        sort_cols.append("qk_tolerance")
        asc.append(True)
    if "quantum_shots" in cand.columns:
        sort_cols.append("quantum_shots")
        asc.append(True)

    if sort_cols:
        pick = cand.sort_values(sort_cols, ascending=asc).iloc[0]
    else:
        pick = cand.iloc[0]

    return {
        "recommended_variant": str(pick.get("variant_name", "")),
        "recommended_tolerance": float(pick["qk_tolerance"]) if "qk_tolerance" in pick else math.nan,
        "recommended_shots": int(pick["quantum_shots"]) if "quantum_shots" in pick and not pd.isna(pick["quantum_shots"]) else -1,
        "recommended_accuracy": float(pick["accuracy_mean"]),
        "recommended_recall10": float(pick["recall_at_10_mean"]) if "recall_at_10_mean" in pick else math.nan,
        "recommended_iters": float(pick["mean_partition_iters_mean"]) if "mean_partition_iters_mean" in pick else math.nan,
        "recommended_train_time": float(pick["train_compress_time_s_mean"]) if "train_compress_time_s_mean" in pick else math.nan,
    }


def recommend_shots(df: pd.DataFrame) -> dict:
    if df.empty or "accuracy_mean" not in df.columns:
        return {}

    exact = df[df["variant_name"] == "quantum_exact"]
    shot = df[df["variant_name"] != "quantum_exact"].copy()
    if shot.empty:
        return {}

    if not exact.empty:
        exact_acc = float(exact.iloc[0]["accuracy_mean"])
        exact_rec = float(exact.iloc[0]["recall_at_10_mean"]) if "recall_at_10_mean" in exact.columns else -np.inf
        cand = shot[shot["accuracy_mean"] >= exact_acc - 0.01].copy()
        if "recall_at_10_mean" in cand.columns:
            cand = cand[cand["recall_at_10_mean"] >= exact_rec - 0.01]
        if cand.empty:
            cand = shot.copy()
    else:
        cand = shot.copy()

    if "distance_fallback_pairs_mean" in cand.columns:
        no_fallback = cand[cand["distance_fallback_pairs_mean"] <= 0]
        if not no_fallback.empty:
            cand = no_fallback

    sort_cols = []
    asc = []
    if "quantum_shots" in cand.columns:
        sort_cols.append("quantum_shots")
        asc.append(True)
    if "train_compress_time_s_mean" in cand.columns:
        sort_cols.append("train_compress_time_s_mean")
        asc.append(True)

    pick = cand.sort_values(sort_cols, ascending=asc).iloc[0] if sort_cols else cand.iloc[0]
    return {
        "recommended_variant": str(pick.get("variant_name", "")),
        "recommended_shots": int(pick["quantum_shots"]) if "quantum_shots" in pick and not pd.isna(pick["quantum_shots"]) else -1,
        "recommended_accuracy": float(pick["accuracy_mean"]),
        "recommended_recall10": float(pick["recall_at_10_mean"]) if "recall_at_10_mean" in pick else math.nan,
        "fallback_warning": bool((df.get("distance_fallback_pairs_mean", pd.Series(dtype=float)) > 0).any()),
    }


def summarize_main_plan(plan_name: str, df: pd.DataFrame, tables_dir: Path, plots_dir: Path) -> dict:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "recall_at_1",
        "recall_at_10",
        "train_compress_time_s",
        "mean_partition_iters",
        "loss_per_point",
    ]
    agg = aggregate_mean_std(df, ["variant_name", "train_size"], metrics)
    save_table(agg, tables_dir / f"{plan_name}_by_train_size.csv")
    save_markdown_table(agg, tables_dir / f"{plan_name}_by_train_size.md")

    for metric in ["accuracy", "recall_at_10", "train_compress_time_s", "mean_partition_iters"]:
        if f"{metric}_mean" in agg.columns:
            plot_line_over_train_size(
                agg,
                metric=metric,
                title=f"{plan_name}: {metric} vs train size",
                out_path=plots_dir / f"{plan_name}_{metric}_vs_train_size.png",
            )

    final_train_size = int(agg["train_size"].max())
    final = agg[agg["train_size"] == final_train_size].copy().sort_values("accuracy_mean", ascending=False)
    save_table(final, tables_dir / f"{plan_name}_final_train_size.csv")
    save_markdown_table(final, tables_dir / f"{plan_name}_final_train_size.md")

    best = final.iloc[0] if not final.empty else None
    return {
        "plan": plan_name,
        "final_train_size": final_train_size,
        "best_variant": str(best["variant_name"]) if best is not None else "",
        "best_accuracy": float(best["accuracy_mean"]) if best is not None else math.nan,
        "best_recall10": float(best["recall_at_10_mean"]) if best is not None and "recall_at_10_mean" in best else math.nan,
    }


def summarize_tolerance_plan(plan_name: str, df: pd.DataFrame, tables_dir: Path, plots_dir: Path) -> dict:
    metrics = [
        "accuracy",
        "recall_at_10",
        "mean_partition_iters",
        "loss_per_point",
        "train_compress_time_s",
        "relative_objective_change_last_mean",
        "accept_ratio_last_mean",
        "backtracks_last_mean",
        "distance_fallback_pairs",
    ]
    agg = aggregate_mean_std(df, ["qk_tolerance", "quantum_shots", "variant_name"], metrics)
    save_table(agg, tables_dir / f"{plan_name}_grid.csv")
    save_markdown_table(agg, tables_dir / f"{plan_name}_grid.md")

    for metric in ["accuracy", "recall_at_10", "mean_partition_iters", "train_compress_time_s", "loss_per_point"]:
        if f"{metric}_mean" in agg.columns:
            plot_tolerance_lines(
                agg,
                metric=metric,
                title=f"{plan_name}: {metric} vs tolerance",
                out_path=plots_dir / f"{plan_name}_{metric}.png",
            )

    rec = recommend_tolerance(agg)
    return {"plan": plan_name, **rec}


def summarize_tolerance_size_variation_plan(plan_name: str, df: pd.DataFrame, tables_dir: Path, plots_dir: Path) -> dict:
    metrics = [
        "accuracy",
        "recall_at_10",
        "mean_partition_iters",
        "loss_per_point",
        "train_compress_time_s",
        "relative_objective_change_last_mean",
        "distance_fallback_pairs",
    ]
    agg = aggregate_mean_std(df, ["variant_name", "train_size", "qk_tolerance", "quantum_shots"], metrics)
    save_table(agg, tables_dir / f"{plan_name}_by_train_size.csv")
    save_markdown_table(agg, tables_dir / f"{plan_name}_by_train_size.md")

    for metric in ["accuracy", "recall_at_10", "mean_partition_iters", "train_compress_time_s"]:
        if f"{metric}_mean" in agg.columns:
            plot_tolerance_size_variation(
                agg,
                metric=metric,
                title=f"{plan_name}: {metric} vs train size",
                out_path=plots_dir / f"{plan_name}_{metric}_vs_train_size.png",
            )

    final_train_size = int(agg["train_size"].max())
    final = agg[agg["train_size"] == final_train_size].copy().sort_values("accuracy_mean", ascending=False)
    save_table(final, tables_dir / f"{plan_name}_final_train_size.csv")
    save_markdown_table(final, tables_dir / f"{plan_name}_final_train_size.md")

    best = final.iloc[0] if not final.empty else None
    return {
        "plan": plan_name,
        "final_train_size": final_train_size,
        "best_variant": str(best["variant_name"]) if best is not None else "",
        "best_tolerance": float(best["qk_tolerance"]) if best is not None else math.nan,
        "best_shots": int(best["quantum_shots"]) if best is not None and not pd.isna(best["quantum_shots"]) else -1,
        "best_accuracy": float(best["accuracy_mean"]) if best is not None else math.nan,
        "best_recall10": float(best["recall_at_10_mean"]) if best is not None and "recall_at_10_mean" in best else math.nan,
    }


def summarize_shots_plan(plan_name: str, df: pd.DataFrame, tables_dir: Path, plots_dir: Path) -> dict:
    metrics = [
        "accuracy",
        "recall_at_10",
        "mean_partition_iters",
        "loss_per_point",
        "train_compress_time_s",
        "distance_fallback_pairs",
    ]
    agg = aggregate_mean_std(df, ["variant_name", "quantum_shots"], metrics)
    save_table(agg, tables_dir / f"{plan_name}_grid.csv")
    save_markdown_table(agg, tables_dir / f"{plan_name}_grid.md")

    for metric in ["accuracy", "recall_at_10", "train_compress_time_s", "mean_partition_iters"]:
        if f"{metric}_mean" in agg.columns:
            plot_shots_lines(
                agg,
                metric=metric,
                title=f"{plan_name}: {metric} vs shots",
                out_path=plots_dir / f"{plan_name}_{metric}.png",
            )

    rec = recommend_shots(agg)
    return {"plan": plan_name, **rec}


def summarize_simple_bar_plan(plan_name: str, df: pd.DataFrame, tables_dir: Path, plots_dir: Path) -> dict:
    metrics = [
        "accuracy",
        "recall_at_10",
        "mean_partition_iters",
        "loss_per_point",
        "train_compress_time_s",
    ]
    agg = aggregate_mean_std(df, ["variant_name"], metrics)
    save_table(agg, tables_dir / f"{plan_name}.csv")
    save_markdown_table(agg, tables_dir / f"{plan_name}.md")

    for metric in ["accuracy", "recall_at_10"]:
        if f"{metric}_mean" in agg.columns:
            plot_bar(
                agg,
                metric=metric,
                title=f"{plan_name}: {metric}",
                out_path=plots_dir / f"{plan_name}_{metric}.png",
            )

    best = agg.sort_values("accuracy_mean", ascending=False).iloc[0] if not agg.empty else None
    return {
        "plan": plan_name,
        "best_variant": str(best["variant_name"]) if best is not None else "",
        "best_accuracy": float(best["accuracy_mean"]) if best is not None else math.nan,
    }


def build_summary_md(items: list[dict], out_path: Path) -> None:
    lines = ["# Paper Experiment Summary", ""]
    for item in items:
        lines.append(f"## {item['plan']}")
        for k, v in item.items():
            if k == "plan":
                continue
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="experiments/paper_runs")
    ap.add_argument("--output-dir", default="experiments/paper_report")
    ap.add_argument(
        "--plans",
        nargs="*",
        default=KNOWN_PLANS,
        help="Subset of plans to analyze; default is all known plans",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    plots_dir, tables_dir, text_dir = ensure_dirs(output_dir)

    loaded = {plan: load_summary(input_dir, plan) for plan in args.plans}
    available = {k: v for k, v in loaded.items() if v is not None}
    if not available:
        raise SystemExit("No summary.csv files found for the selected plans.")

    all_runs = pd.concat(list(available.values()), ignore_index=True)
    all_runs.to_csv(output_dir / "all_runs.csv", index=False)

    global_metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "recall_at_1",
        "recall_at_10",
        "train_compress_time_s",
        "mean_partition_iters",
        "loss_per_point",
    ]
    global_agg = aggregate_mean_std(
        all_runs,
        ["plan", "variant_name", "train_size"],
        global_metrics,
    )
    save_table(global_agg, tables_dir / "all_plans_aggregate.csv")
    save_markdown_table(global_agg, tables_dir / "all_plans_aggregate.md")

    summaries = []

    if "main_digits" in available:
        summaries.append(summarize_main_plan("main_digits", available["main_digits"], tables_dir, plots_dir))
    if "main_fashion" in available:
        summaries.append(summarize_main_plan("main_fashion", available["main_fashion"], tables_dir, plots_dir))
    if "ablate_tolerance_digits" in available:
        summaries.append(summarize_tolerance_plan("ablate_tolerance_digits", available["ablate_tolerance_digits"], tables_dir, plots_dir))
    if "ablate_tolerance_high_digits" in available:
        summaries.append(summarize_tolerance_plan("ablate_tolerance_high_digits", available["ablate_tolerance_high_digits"], tables_dir, plots_dir))
    if "ablate_tolerance_high_digits_size_variation" in available:
        summaries.append(summarize_tolerance_size_variation_plan("ablate_tolerance_high_digits_size_variation", available["ablate_tolerance_high_digits_size_variation"], tables_dir, plots_dir))
    #SO AUSWERTEN: lol
    #python make_paper_report.py --plans main_fashion ablate_fashion_high_shots_size_variation
    if "ablate_fashion_high_shots_size_variation" in available:
        summaries.append(
            summarize_tolerance_size_variation_plan(
                "ablate_fashion_high_shots_size_variation",
                available["ablate_fashion_high_shots_size_variation"],
                tables_dir,
                plots_dir,
            )
        )
    if "ablate_shots_digits" in available:
        summaries.append(summarize_shots_plan("ablate_shots_digits", available["ablate_shots_digits"], tables_dir, plots_dir))
    if "ablate_metric_digits" in available:
        summaries.append(summarize_simple_bar_plan("ablate_metric_digits", available["ablate_metric_digits"], tables_dir, plots_dir))
    if "ablate_signaware_synth" in available:
        summaries.append(summarize_simple_bar_plan("ablate_signaware_synth", available["ablate_signaware_synth"], tables_dir, plots_dir))

    build_summary_md(summaries, text_dir / "paper_summary.md")
    (text_dir / "paper_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[INFO] Report written to {output_dir}")


if __name__ == "__main__":
    main()