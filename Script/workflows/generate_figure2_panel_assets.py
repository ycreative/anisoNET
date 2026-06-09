"""Generate separated Figure 2 panel assets for GSE193107 primary application."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from PIL import Image


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ROOT = Path(os.environ.get("ANISONET_ANALYSIS_ROOT", PROJECT_ROOT / "codexAnalysis"))
OUT_DIR = ROOT / "manuscript_figures" / "Figure2_gse193107_primary_application"
PREFLIGHT_ROOT = ROOT / "preflight" / "brain_aging_gse193107"
PINN_ROOT = ROOT / "pinn" / "brain_aging_gse193107"
SPATIAL_ROOT = ROOT / "processed_visium" / "brain_aging_gse193107"
BATCH_ROOT = ROOT / "batch" / "brain_aging_gse193107"
METRIC_SUMMARY = ROOT / "barrier_field_metrics" / "barrier_field_metrics_summary.csv"

REP_SAMPLE = "GSM5773457_Old_mouse_brain_A1-2"
REP_TASK = "Apoe_CNS_Myelin"
PINN_RUN = "fourier_refined_16g_gauss07_batch"
TARGETS = {"Apoe": "Apoe_CNS_Myelin", "Gfap": "Gfap_CNS_Myelin"}


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_panel(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    ensure_dir(OUT_DIR)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.08) -> None:
    return None


def sample_label(sample: str) -> str:
    condition = "Young" if "_Young_" in sample else "Old"
    replicate = sample.split("_brain_")[-1]
    gsm = sample.split("_")[0].replace("GSM577", "G")
    return f"{condition} {replicate}\n{gsm}"


def crop_to_mask(arr: np.ndarray, mask: np.ndarray, pad: int = 5) -> np.ndarray:
    yy, xx = np.where(mask)
    if yy.size == 0:
        return arr
    y0 = max(int(yy.min()) - pad, 0)
    y1 = min(int(yy.max()) + pad + 1, arr.shape[0])
    x0 = max(int(xx.min()) - pad, 0)
    x1 = min(int(xx.max()) + pad + 1, arr.shape[1])
    return arr[y0:y1, x0:x1]


def normalized_panel(arr: np.ndarray, mask: np.ndarray | None = None, *, log: bool = False) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    if log:
        finite = data[np.isfinite(data)]
        floor = np.nanpercentile(finite, 1) * 0.1 if finite.size else 1e-6
        data = np.log10(np.maximum(data, floor))
    valid = np.isfinite(data)
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        return np.zeros_like(data)
    lo, hi = np.nanpercentile(data[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((data - lo) / (hi - lo), 0, 1)
    if mask is not None:
        out = np.where(mask, out, np.nan)
    return out


def show_grid(ax: plt.Axes, arr: np.ndarray, mask: np.ndarray, title: str, cmap: str, *, log: bool = False) -> None:
    panel = crop_to_mask(normalized_panel(arr, mask, log=log), mask)
    ax.imshow(panel, cmap=cmap, interpolation="bilinear", origin="lower")
    ax.set_title(title, fontsize=7.4, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def show_he_with_spots(ax: plt.Axes, sample: str, coords_norm: np.ndarray) -> None:
    image_path = SPATIAL_ROOT / sample / "spatial" / "tissue_hires_image.png"
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((900, 900), Image.Resampling.LANCZOS)
    arr = np.asarray(image)
    ax.imshow(arr)
    h, w = arr.shape[:2]
    x = np.clip(coords_norm[:, 0], 0, 1) * w
    y = (1 - np.clip(coords_norm[:, 1], 0, 1)) * h
    ax.scatter(x, y, s=3.0, facecolor="#f08a4b", edgecolor="white", linewidth=0.10, alpha=0.72)
    ax.set_title("H&E + Visium spots", fontsize=7.4, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig2a_dataset_design() -> None:
    fig, ax = plt.subplots(figsize=(3.7, 2.15))
    ax.axis("off")
    panel_label(ax, "A", x=-0.02, y=1.03)
    ax.text(0.06, 0.88, "GSE193107 aging mouse brain", fontsize=10, fontweight="bold", color="#263238")
    rows = [
        ("Sections", "8 Visium sections"),
        ("Conditions", "4 young, 4 old"),
        ("Targets", "Apoe, Gfap"),
        ("Barrier prior", "CNS-myelin: Mbp, Plp1, Mobp"),
        ("Model profile", "scalar PINN, gauss07 postprocess"),
        ("Primary task", "tissue-constrained source-field inference"),
    ]
    y = 0.72
    for key, value in rows:
        ax.text(0.08, y, key, fontsize=7.4, fontweight="bold", color="#35424d")
        ax.text(0.42, y, value, fontsize=7.4, color="#35424d")
        y -= 0.105
    ax.text(0.06, 0.09, "Claim boundary: not a universal interpolation-superiority benchmark.", fontsize=6.7, color="#8a1c1c")
    save_panel(fig, "Fig2A_dataset_design")


def fig2b_representative_input_stack() -> None:
    preflight = PREFLIGHT_ROOT / REP_SAMPLE / REP_TASK
    pinn = PINN_ROOT / REP_SAMPLE / REP_TASK / PINN_RUN
    mask = np.load(preflight / "tissue_mask.npy") > 0
    coords = np.load(preflight / "coords_norm.npy")
    source = np.load(preflight / "source_grid.npy")
    barrier = np.load(preflight / "barrier_grid.npy")
    resistance = np.load(preflight / "resistance_grid.npy")
    diffusion = np.load(preflight / "diffusion_grid.npy")
    field = np.load(pinn / "pinn_grid_prediction_postprocessed.npy")

    fig = plt.figure(figsize=(7.4, 2.0))
    gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.12)
    axes = [fig.add_subplot(gs[0, i]) for i in range(6)]
    panel_label(axes[0], "B", x=-0.18, y=1.08)
    show_he_with_spots(axes[0], REP_SAMPLE, coords)
    show_grid(axes[1], source, mask, "Source\nApoe", "magma")
    show_grid(axes[2], barrier, mask, "Barrier\nCNS myelin", "YlOrBr")
    show_grid(axes[3], resistance, mask, "Resistance\nR=1/D", "viridis", log=True)
    show_grid(axes[4], diffusion, mask, "Diffusion\nD(x,y)", "cividis")
    show_grid(axes[5], field, mask, "Postprocessed\nfield", "plasma")
    fig.suptitle("Representative old A1 input and output stack", fontsize=9, fontweight="bold", y=1.03)
    save_panel(fig, "Fig2B_representative_input_stack")


def load_manifest(gene: str) -> list[str]:
    path = BATCH_ROOT / f"{gene}_CNS_Myelin" / "batch_manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return list(json.load(handle)["samples"])


def fig2_field_montage(gene: str, panel: str) -> None:
    task = TARGETS[gene]
    samples = load_manifest(gene)
    metrics = pd.read_csv(BATCH_ROOT / task / "batch_metrics_summary.csv")
    metrics = metrics[metrics["field_type"] == "gauss07"].set_index("sample")

    fields: list[tuple[str, np.ndarray]] = []
    tissue_values = []
    for sample in samples:
        field = np.load(PINN_ROOT / sample / task / PINN_RUN / "pinn_grid_prediction_postprocessed.npy")
        mask = np.load(PREFLIGHT_ROOT / sample / task / "tissue_mask.npy") > 0
        fields.append((sample, crop_to_mask(field, mask)))
        tissue_values.append(field[mask].reshape(-1))
    vmax = float(np.percentile(np.concatenate(tissue_values), 99.5))

    fig = plt.figure(figsize=(7.2, 3.15))
    gs = gridspec.GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.035], wspace=0.05, hspace=0.36)
    image = None
    for idx, (sample, field) in enumerate(fields):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        if idx == 0:
            panel_label(ax, panel, x=-0.18, y=1.15)
        image = ax.imshow(field, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        r = float(metrics.loc[sample, "spot_pearson_source"])
        ax.set_title(f"{sample_label(sample)}\nr={r:.3f}", fontsize=5.8, pad=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_subplot(gs[:, 4])
    cbar = fig.colorbar(image, cax=cax)
    cbar.outline.set_linewidth(0.3)
    cbar.ax.tick_params(labelsize=5.5, length=1.5, width=0.3)
    fig.suptitle(f"{gene} tissue-constrained fields across 8 sections", fontsize=9, fontweight="bold", y=1.02)
    save_panel(fig, f"Fig2{panel}_{gene}_8section_fields")


def fig2e_source_fidelity_roughness() -> None:
    df = pd.read_csv(BATCH_ROOT / "target_batch_comparison.csv")
    df = df[df["field_type"] == "gauss07"].copy()
    df["label"] = df["target"] + "\n" + df["condition"]
    colors = df["target"].map({"Apoe": "#4C78A8", "Gfap": "#B279A2"})
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(5.7, 2.35), gridspec_kw={"wspace": 0.32})
    panel_label(axes[0], "E", x=-0.16, y=1.10)
    axes[0].bar(x, df["pearson_mean"], yerr=df["pearson_sem"], color=colors, edgecolor="#222222", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["label"], fontsize=6.8)
    axes[0].set_ylim(0.55, 0.95)
    axes[0].set_ylabel("Source-field Pearson")
    axes[0].set_title("Source fidelity")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.5)

    axes[1].bar(x, df["roughness_p95_mean"], yerr=df["roughness_p95_sem"], color=colors, edgecolor="#222222", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["label"], fontsize=6.8)
    axes[1].set_ylabel("Gradient p95")
    axes[1].set_title("Field roughness")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.5)
    fig.suptitle("Field agreement and smoothness across GSE193107", fontsize=9, fontweight="bold", y=1.04)
    save_panel(fig, "Fig2E_source_fidelity_and_roughness")


def fig2f_tissue_support_metrics() -> None:
    summary = pd.read_csv(METRIC_SUMMARY)
    leakage = summary[
        (summary["dataset"] == "GSE193107")
        & summary["field_type"].isin(["raw", "postprocessed"])
        & (summary["metric"] == "tissue_support_leakage")
    ].copy()
    leakage["label"] = leakage["target"] + "\n" + leakage["condition"]
    order = ["Apoe\nYoung", "Apoe\nOld", "Gfap\nYoung", "Gfap\nOld"]
    x_base = np.arange(len(order))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.35), gridspec_kw={"wspace": 0.34})
    panel_label(axes[0], "F", x=-0.16, y=1.10)
    for i, field_type in enumerate(["raw", "postprocessed"]):
        sub = leakage[leakage["field_type"] == field_type].set_index("label").reindex(order)
        axes[0].bar(
            x_base + (i - 0.5) * width,
            sub["value_mean"],
            yerr=sub["value_sem"],
            width=width,
            label=field_type,
            color={"raw": "#D95F02", "postprocessed": "#1B9E77"}[field_type],
            edgecolor="#222222",
            linewidth=0.5,
        )
    axes[0].set_xticks(x_base)
    axes[0].set_xticklabels(order, fontsize=6.8)
    axes[0].set_ylabel("Tissue-support leakage")
    axes[0].set_title("Background field mass")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper right")

    reduction = summary[
        (summary["dataset"] == "GSE193107")
        & (summary["field_type"] == "raw_to_postprocessed")
        & (summary["metric"] == "tissue_mask_leakage_reduction_index")
    ].copy()
    reduction["label"] = reduction["target"] + "\n" + reduction["condition"]
    sub = reduction.set_index("label").reindex(order)
    axes[1].bar(x_base, sub["value_mean"], yerr=sub["value_sem"], color="#5E81AC", edgecolor="#222222", linewidth=0.5)
    axes[1].set_xticks(x_base)
    axes[1].set_xticklabels(order, fontsize=6.8)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Leakage reduction index")
    axes[1].set_title("Raw to postprocessed")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.5)
    fig.suptitle("Tissue-mask constraint of output fields", fontsize=9, fontweight="bold", y=1.04)
    save_panel(fig, "Fig2F_tissue_support_leakage")


def fig2g_barrier_context_summary() -> None:
    summary = pd.read_csv(METRIC_SUMMARY)
    sub = summary[
        (summary["dataset"] == "GSE193107")
        & (summary["field_type"] == "postprocessed")
        & (summary["metric"] == "grid_high_to_low_barrier_ratio")
    ].copy()
    sub["label"] = sub["target"] + "\n" + sub["condition"]
    order = ["Apoe\nYoung", "Apoe\nOld", "Gfap\nYoung", "Gfap\nOld"]
    sub = sub.set_index("label").reindex(order)
    colors = ["#4C78A8", "#4C78A8", "#B279A2", "#B279A2"]
    fig, ax = plt.subplots(figsize=(3.6, 2.35))
    panel_label(ax, "G", x=-0.16, y=1.10)
    x = np.arange(len(order))
    ax.bar(x, sub["value_mean"], yerr=sub["value_sem"], color=colors, edgecolor="#222222", linewidth=0.5)
    ax.axhline(1.0, color="#555555", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=6.8)
    ax.set_ylabel("High/low barrier field ratio")
    ax.set_title("Barrier-context field summary")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.text(0.02, 0.04, "Descriptive for real tissue;\nnot a universal attenuation claim.", transform=ax.transAxes, fontsize=6.4, color="#555555")
    save_panel(fig, "Fig2G_barrier_context_summary")


def rough_assembly() -> None:
    panel_files = [
        "Fig2A_dataset_design.png",
        "Fig2B_representative_input_stack.png",
        "Fig2C_Apoe_8section_fields.png",
        "Fig2D_Gfap_8section_fields.png",
        "Fig2E_source_fidelity_and_roughness.png",
        "Fig2F_tissue_support_leakage.png",
        "Fig2G_barrier_context_summary.png",
    ]
    fig = plt.figure(figsize=(8.4, 11.2))
    gs = gridspec.GridSpec(6, 2, figure=fig, height_ratios=[0.70, 0.82, 1.10, 1.10, 0.92, 0.72], hspace=0.20, wspace=0.12)
    layout = [
        (panel_files[0], gs[0, 0]),
        (panel_files[1], gs[0, 1]),
        (panel_files[2], gs[1:3, :]),
        (panel_files[3], gs[3:5, :]),
        (panel_files[4], gs[5, 0]),
        (panel_files[5], gs[5, 1]),
    ]
    # The rough assembly is intentionally imperfect; individual panel files are the source assets.
    for file_name, spec in layout:
        ax = fig.add_subplot(spec)
        img = Image.open(OUT_DIR / file_name).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle("Figure 2 rough assembly: GSE193107 primary application", fontsize=12, fontweight="bold", y=0.99)
    save_panel(fig, "Figure2_rough_assembly")


def main() -> None:
    setup()
    ensure_dir(OUT_DIR)
    fig2a_dataset_design()
    fig2b_representative_input_stack()
    fig2_field_montage("Apoe", "C")
    fig2_field_montage("Gfap", "D")
    fig2e_source_fidelity_roughness()
    fig2f_tissue_support_metrics()
    fig2g_barrier_context_summary()
    rough_assembly()
    print(f"Wrote Figure 2 panel assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
