"""Analyze train-fitted blends of kidney prior and prior-hybrid PINN fields."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "Script"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anisonet.metrics import mse, pearsonr_safe, sample_grid_at_spots
from anisonet.preprocessing import clip_and_normalize
from anisonet.visium_io import load_visium_lite, normalized_gene_vector


PREFLIGHT_BY_TARGET = {
    "Umod": "Umod_proximal_barrier",
    "Slc34a1": "Slc34a1_TAL_CD_barrier",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit readout blends for kidney prior-hybrid fields.")
    parser.add_argument(
        "--sweep-root",
        default="codexAnalysis/barrier_split_anisonet/mouse_kidney_10x/V1_Mouse_Kidney/prior_hybrid_sweep",
    )
    parser.add_argument(
        "--sample-dir",
        default="codexAnalysis/processed_visium/mouse_kidney_10x/V1_Mouse_Kidney",
    )
    parser.add_argument(
        "--preflight-root",
        default="codexAnalysis/cross_tissue/mouse_kidney_10x/V1_Mouse_Kidney/preflight",
    )
    parser.add_argument(
        "--output-dir",
        default="codexAnalysis/barrier_split_anisonet/mouse_kidney_10x/V1_Mouse_Kidney/prior_hybrid_blend_analysis",
    )
    parser.add_argument("--targets", default="Umod,Slc34a1")
    parser.add_argument("--prior-weights", default="5,10,20,40")
    parser.add_argument("--seeds", default="0,1,2")
    return parser.parse_args()


def parse_csv(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def fit_affine(train_pred: np.ndarray, train_truth: np.ndarray) -> tuple[float, float]:
    x = np.asarray(train_pred, dtype=np.float64).reshape(-1)
    y = np.asarray(train_truth, dtype=np.float64).reshape(-1)
    var = float(np.var(x))
    if var <= 1e-12:
        return 0.0, float(np.mean(y))
    slope = float(np.cov(x, y, bias=True)[0, 1] / var)
    intercept = float(np.mean(y) - slope * np.mean(x))
    return slope, intercept


def apply_affine(values: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return np.clip(slope * np.asarray(values, dtype=np.float64) + intercept, 0.0, 1.0).astype(np.float32)


def fit_linear2(train_prior: np.ndarray, train_pinn: np.ndarray, train_truth: np.ndarray, *, ridge: float = 1e-6) -> tuple[float, float, float]:
    x = np.column_stack(
        [
            np.asarray(train_prior, dtype=np.float64).reshape(-1),
            np.asarray(train_pinn, dtype=np.float64).reshape(-1),
            np.ones_like(train_truth, dtype=np.float64).reshape(-1),
        ]
    )
    y = np.asarray(train_truth, dtype=np.float64).reshape(-1)
    penalty = np.diag([ridge, ridge, 0.0])
    coefs = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return float(coefs[0]), float(coefs[1]), float(coefs[2])


def apply_linear2(prior: np.ndarray, pinn: np.ndarray, coef_prior: float, coef_pinn: float, intercept: float) -> np.ndarray:
    values = coef_prior * np.asarray(prior, dtype=np.float64) + coef_pinn * np.asarray(pinn, dtype=np.float64) + intercept
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def fit_convex_lambda(
    train_prior: np.ndarray,
    train_pinn: np.ndarray,
    train_truth: np.ndarray,
    *,
    grid_size: int = 101,
) -> float:
    lambdas = np.linspace(0.0, 1.0, grid_size)
    best_lambda = 0.0
    best_mse = float("inf")
    for lam in lambdas:
        pred = (1.0 - lam) * train_prior + lam * train_pinn
        value = mse(pred, train_truth)
        if value < best_mse:
            best_mse = value
            best_lambda = float(lam)
    return best_lambda


def metric_row(
    *,
    target_gene: str,
    prior_weight: float,
    seed: int,
    method: str,
    pred_train: np.ndarray,
    pred_test: np.ndarray,
    train_truth: np.ndarray,
    test_truth: np.ndarray,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "target_gene": target_gene,
        "prior_weight": prior_weight,
        "seed": seed,
        "method": method,
        "train_pearson": pearsonr_safe(pred_train, train_truth),
        "train_mse": mse(pred_train, train_truth),
        "test_pearson": pearsonr_safe(pred_test, test_truth),
        "test_mse": mse(pred_test, test_truth),
        "prediction_mean_test": float(np.mean(pred_test)),
        "prediction_sd_test": float(np.std(pred_test)),
        **(extra or {}),
    }


def run_one(
    *,
    sample,
    target_gene: str,
    preflight_root: Path,
    run_dir: Path,
    prior_weight: float,
    seed: int,
) -> list[dict[str, object]]:
    preflight_dir = preflight_root / PREFLIGHT_BY_TARGET[target_gene]
    coords_norm = np.load(preflight_dir / "coords_norm.npy")
    source_values = clip_and_normalize(normalized_gene_vector(sample, target_gene), percentile=99.0)

    train_idx = np.load(run_dir / "train_idx.npy")
    test_idx = np.load(run_dir / "test_idx.npy")
    train_coords = coords_norm[train_idx]
    test_coords = coords_norm[test_idx]
    train_truth = source_values[train_idx]
    test_truth = source_values[test_idx]

    prior_grid = np.load(run_dir / "prior_line_resistance_idw_grid.npy")
    pinn_grid = np.load(run_dir / "anisonet_gauss07_grid.npy")
    prior_train = sample_grid_at_spots(prior_grid, train_coords)
    prior_test = sample_grid_at_spots(prior_grid, test_coords)
    pinn_train = sample_grid_at_spots(pinn_grid, train_coords)
    pinn_test = sample_grid_at_spots(pinn_grid, test_coords)

    rows = []
    rows.append(
        metric_row(
            target_gene=target_gene,
            prior_weight=prior_weight,
            seed=seed,
            method="prior_grid_raw",
            pred_train=prior_train,
            pred_test=prior_test,
            train_truth=train_truth,
            test_truth=test_truth,
        )
    )

    slope, intercept = fit_affine(prior_train, train_truth)
    rows.append(
        metric_row(
            target_gene=target_gene,
            prior_weight=prior_weight,
            seed=seed,
            method="prior_grid_traincal",
            pred_train=apply_affine(prior_train, slope, intercept),
            pred_test=apply_affine(prior_test, slope, intercept),
            train_truth=train_truth,
            test_truth=test_truth,
            extra={"slope": slope, "intercept": intercept},
        )
    )

    slope, intercept = fit_affine(pinn_train, train_truth)
    rows.append(
        metric_row(
            target_gene=target_gene,
            prior_weight=prior_weight,
            seed=seed,
            method="pinn_gauss07_traincal",
            pred_train=apply_affine(pinn_train, slope, intercept),
            pred_test=apply_affine(pinn_test, slope, intercept),
            train_truth=train_truth,
            test_truth=test_truth,
            extra={"slope": slope, "intercept": intercept},
        )
    )

    lam = fit_convex_lambda(prior_train, pinn_train, train_truth)
    convex_train = (1.0 - lam) * prior_train + lam * pinn_train
    convex_test = (1.0 - lam) * prior_test + lam * pinn_test
    rows.append(
        metric_row(
            target_gene=target_gene,
            prior_weight=prior_weight,
            seed=seed,
            method="prior_pinn_convex_trainfit",
            pred_train=convex_train,
            pred_test=convex_test,
            train_truth=train_truth,
            test_truth=test_truth,
            extra={"lambda_pinn": lam},
        )
    )

    slope, intercept = fit_affine(convex_train, train_truth)
    rows.append(
        metric_row(
            target_gene=target_gene,
            prior_weight=prior_weight,
            seed=seed,
            method="prior_pinn_convex_trainfit_traincal",
            pred_train=apply_affine(convex_train, slope, intercept),
            pred_test=apply_affine(convex_test, slope, intercept),
            train_truth=train_truth,
            test_truth=test_truth,
            extra={"lambda_pinn": lam, "slope": slope, "intercept": intercept},
        )
    )

    coef_prior, coef_pinn, linear_intercept = fit_linear2(prior_train, pinn_train, train_truth)
    rows.append(
        metric_row(
            target_gene=target_gene,
            prior_weight=prior_weight,
            seed=seed,
            method="prior_pinn_linear2_trainfit",
            pred_train=apply_linear2(prior_train, pinn_train, coef_prior, coef_pinn, linear_intercept),
            pred_test=apply_linear2(prior_test, pinn_test, coef_prior, coef_pinn, linear_intercept),
            train_truth=train_truth,
            test_truth=test_truth,
            extra={"coef_prior": coef_prior, "coef_pinn": coef_pinn, "intercept": linear_intercept},
        )
    )
    return rows


def summarize(output_dir: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "kidney_prior_hybrid_blend_metrics.csv", index=False)

    baseline = frame[frame["method"] == "prior_grid_raw"][
        ["target_gene", "prior_weight", "seed", "test_pearson", "test_mse"]
    ].rename(columns={"test_pearson": "prior_grid_pearson", "test_mse": "prior_grid_mse"})
    with_baseline = frame.merge(baseline, on=["target_gene", "prior_weight", "seed"], how="left")
    with_baseline["pearson_delta_vs_prior_grid"] = with_baseline["test_pearson"] - with_baseline["prior_grid_pearson"]
    with_baseline["mse_delta_vs_prior_grid"] = with_baseline["test_mse"] - with_baseline["prior_grid_mse"]
    with_baseline.to_csv(output_dir / "kidney_prior_hybrid_blend_deltas.csv", index=False)

    summary = (
        with_baseline.groupby(["target_gene", "method"], as_index=False)
        .agg(
            n=("test_pearson", "size"),
            test_pearson_mean=("test_pearson", "mean"),
            test_pearson_sd=("test_pearson", "std"),
            test_mse_mean=("test_mse", "mean"),
            test_mse_sd=("test_mse", "std"),
            pearson_delta_vs_prior_grid_mean=("pearson_delta_vs_prior_grid", "mean"),
            pearson_delta_vs_prior_grid_n_positive=("pearson_delta_vs_prior_grid", lambda x: int((x > 0).sum())),
            mse_delta_vs_prior_grid_mean=("mse_delta_vs_prior_grid", "mean"),
            mse_delta_vs_prior_grid_n_negative=("mse_delta_vs_prior_grid", lambda x: int((x < 0).sum())),
        )
    )
    summary.to_csv(output_dir / "kidney_prior_hybrid_blend_summary.csv", index=False)

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    plot_methods = [
        "prior_grid_raw",
        "pinn_gauss07_traincal",
        "prior_pinn_convex_trainfit_traincal",
        "prior_pinn_linear2_trainfit",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), constrained_layout=True)
    for ax, metric, ylabel in [
        (axes[0], "test_pearson", "Held-out Pearson"),
        (axes[1], "test_mse", "Held-out MSE"),
    ]:
        pivot = summary[summary["method"].isin(plot_methods)].pivot_table(
            index="target_gene", columns="method", values=f"{metric}_mean", aggfunc="first"
        )
        pivot = pivot.reindex(columns=[method for method in plot_methods if method in pivot.columns])
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=0)
        ax.legend(fontsize=6, frameon=False)
    fig.savefig(output_dir / "kidney_prior_hybrid_blend_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "kidney_prior_hybrid_blend_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Kidney Prior-Hybrid Blend Analysis",
        "",
        "This analysis uses existing prior-hybrid sweep outputs and fits readout-level combinations on training spots only.",
        "",
        "## Summary",
        "",
    ]
    for target in sorted(summary["target_gene"].unique()):
        target_summary = summary[summary["target_gene"] == target].sort_values("test_pearson_mean", ascending=False)
        top = target_summary.iloc[0]
        best_mse = target_summary.sort_values("test_mse_mean", ascending=True).iloc[0]
        lines.append(
            f"- `{target}` best Pearson method: `{top.method}` with mean Pearson `{top.test_pearson_mean:.4f}` "
            f"and mean MSE `{top.test_mse_mean:.5f}`."
        )
        lines.append(
            f"- `{target}` best MSE method: `{best_mse.method}` with mean Pearson `{best_mse.test_pearson_mean:.4f}` "
            f"and mean MSE `{best_mse.test_mse_mean:.5f}`."
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "If a train-fitted blend improves Pearson without increasing MSE relative to the raw continuous prior, it is a candidate formal readout. If not, the PINN is still disturbing useful amplitude information from the prior."
    )
    (output_dir / "kidney_prior_hybrid_blend_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = load_visium_lite(args.sample_dir)
    targets = parse_csv(args.targets, str)
    weights = parse_csv(args.prior_weights, float)
    seeds = parse_csv(args.seeds, int)

    rows: list[dict[str, object]] = []
    for target in targets:
        for weight in weights:
            for seed in seeds:
                run_dir = Path(args.sweep_root) / target / f"priorw{weight:g}_seed{seed}"
                rows.extend(
                    run_one(
                        sample=sample,
                        target_gene=target,
                        preflight_root=Path(args.preflight_root),
                        run_dir=run_dir,
                        prior_weight=weight,
                        seed=seed,
                    )
                )
    summarize(output_dir, rows)
    print(f"Wrote kidney prior-hybrid blend analysis to {output_dir}")


if __name__ == "__main__":
    main()

