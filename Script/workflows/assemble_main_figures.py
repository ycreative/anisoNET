"""Assemble publication-style main figure drafts from existing anisoNET outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from PIL import Image, ImageChops


ROOT = Path("codexAnalysis")
OUT_DIR = ROOT / "manuscript_figures" / "main"
PREFLIGHT_ROOT = ROOT / "preflight" / "brain_aging_gse193107"
PINN_ROOT = ROOT / "pinn" / "brain_aging_gse193107"
SYNTHETIC_ROOT = (
    ROOT
    / "synthetic_barrier"
    / "brain_aging_gse193107"
    / "GSM5773457_Old_mouse_brain_A1-2"
    / "Apoe_CNS_Myelin"
)


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "savefig.dpi": 600,
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def crop_white_border(path: Path, pad: int = 12, top_extra: int = 0) -> Image.Image:
    image = Image.open(path).convert("RGB")
    bg = Image.new("RGB", image.size, (255, 255, 255))
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    left = max(left - pad, 0)
    top = max(top - pad, 0)
    right = min(right + pad, image.width)
    bottom = min(bottom + pad, image.height)
    top = min(top + top_extra, bottom - 1)
    return image.crop((left, top, right, bottom))


def panel_label(ax: plt.Axes, label: str, x: float = -0.04, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    ensure_dir(OUT_DIR)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def draw_metric_summary(ax: plt.Axes) -> None:
    paths = {
        "Apoe": ROOT / "batch" / "brain_aging_gse193107" / "Apoe_CNS_Myelin" / "group_summary.csv",
        "Gfap": ROOT / "batch" / "brain_aging_gse193107" / "Gfap_CNS_Myelin" / "group_summary.csv",
    }
    rows = []
    for gene, path in paths.items():
        df = pd.read_csv(path)
        sub = df[df["field_type"] == "gauss07"].copy()
        for _, row in sub.iterrows():
            rows.append(
                {
                    "gene": gene,
                    "condition": row["condition"],
                    "pearson": row["spot_pearson_source_mean"],
                    "pearson_sem": row["spot_pearson_source_sem"],
                    "roughness": row["roughness_grad_p95_mean"],
                    "roughness_sem": row["roughness_grad_p95_sem"],
                }
            )
    summary = pd.DataFrame(rows)
    x = np.arange(len(summary))
    colors = summary["gene"].map({"Apoe": "#4C78A8", "Gfap": "#B279A2"}).tolist()
    ax.bar(x, summary["pearson"], yerr=summary["pearson_sem"], color=colors, edgecolor="#222222", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n{c}" for g, c in zip(summary["gene"], summary["condition"])], rotation=0)
    ax.set_ylabel("Source-field Pearson")
    ax.set_ylim(0, 1.0)
    ax.set_title("Field agreement across 8 sections", fontsize=9, pad=4)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)


def sample_label(sample: str) -> str:
    condition = "Young" if "_Young_" in sample else "Old"
    replicate = sample.split("_brain_")[-1]
    gsm = sample.split("_")[0].replace("GSM577", "G")
    return f"{condition} {replicate}\n{gsm}"


def crop_to_mask(field: np.ndarray, mask: np.ndarray, pad: int = 4) -> np.ndarray:
    yy, xx = np.where(mask)
    if yy.size == 0:
        return field
    y0 = max(int(yy.min()) - pad, 0)
    y1 = min(int(yy.max()) + pad + 1, field.shape[0])
    x0 = max(int(xx.min()) - pad, 0)
    x1 = min(int(xx.max()) + pad + 1, field.shape[1])
    return field[y0:y1, x0:x1]


def field_grid(fig: plt.Figure, spec: gridspec.SubplotSpec, gene: str, panel: str) -> None:
    manifest_path = ROOT / "batch" / "brain_aging_gse193107" / f"{gene}_CNS_Myelin" / "batch_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    samples = list(manifest["samples"])
    metrics = pd.read_csv(ROOT / "batch" / "brain_aging_gse193107" / f"{gene}_CNS_Myelin" / "batch_metrics_summary.csv")
    metrics = metrics[metrics["field_type"] == "gauss07"].set_index("sample")
    target_name = f"{gene}_CNS_Myelin"
    run_name = "fourier_refined_16g_gauss07_batch"

    fields: list[tuple[str, np.ndarray]] = []
    tissue_values = []
    for sample in samples:
        field = np.load(PINN_ROOT / sample / target_name / run_name / "pinn_grid_prediction_postprocessed.npy")
        mask = np.load(PREFLIGHT_ROOT / sample / target_name / "tissue_mask.npy")
        crop = crop_to_mask(field, mask)
        fields.append((sample, crop))
        tissue_values.append(field[mask].reshape(-1))
    vmax = float(np.percentile(np.concatenate(tissue_values), 99.5))

    inner = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=spec, width_ratios=[1, 1, 1, 1, 0.035], wspace=0.06, hspace=0.42)
    image = None
    axes: list[plt.Axes] = []
    for idx, (sample, field) in enumerate(fields):
        ax = fig.add_subplot(inner[idx // 4, idx % 4])
        axes.append(ax)
        image = ax.imshow(field, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        r = float(metrics.loc[sample, "spot_pearson_source"])
        ax.set_title(f"{sample_label(sample)}\nr={r:.3f}", fontsize=5.2, pad=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    cax = fig.add_subplot(inner[:, 4])
    cbar = fig.colorbar(image, cax=cax)
    cbar.outline.set_linewidth(0.3)
    cbar.ax.tick_params(labelsize=5, length=1.5, width=0.3)

    label_ax = fig.add_subplot(spec)
    label_ax.axis("off")
    label_ax.text(-0.03, 1.04, panel, transform=label_ax.transAxes, fontsize=13, fontweight="bold", ha="left", va="top")
    label_ax.text(-0.03, 0.93, gene, transform=label_ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="top")


def assemble_figure2() -> None:
    setup_matplotlib()

    fig = plt.figure(figsize=(7.2, 7.8), constrained_layout=False)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.5, 1.72, 1.72], hspace=0.52, wspace=0.28)

    ax_info = fig.add_subplot(gs[0, 0])
    ax_info.axis("off")
    panel_label(ax_info, "A", x=-0.02, y=1.03)
    ax_info.text(0.05, 0.85, "GSE193107 brain aging Visium", fontsize=9.5, fontweight="bold", ha="left")
    ax_info.text(
        0.05,
        0.55,
        "8 sections: 4 young, 4 old\nTargets: Apoe and Gfap\nBarrier module: Mbp, Plp1, Mobp\nProfile: conservative scalar PINN + tissue mask",
        fontsize=8,
        ha="left",
        va="top",
        linespacing=1.25,
    )

    ax_metric = fig.add_subplot(gs[0, 1])
    panel_label(ax_metric, "B", x=-0.14, y=1.08)
    draw_metric_summary(ax_metric)

    field_grid(fig, gs[1, :], "Apoe", "C")
    field_grid(fig, gs[2, :], "Gfap", "D")

    fig.suptitle("anisoNET tissue-constrained source fields in GSE193107", fontsize=11, fontweight="bold", y=0.985)
    save_figure(fig, "Figure2_gse193107_field_reconstruction_draft")


def clean_method_label(label: str) -> str:
    mapping = {
        "nearest": "Nearest",
        "idw_k8": "IDW",
        "gaussian_sigma1p5": "Gaussian 1.5",
        "gaussian_sigma3": "Gaussian 3",
        "anisoNET_masked": "anisoNET\nmasked",
        "anisoNET_gauss07": "anisoNET\nsmooth",
        "anisoNET_original_barrier": "anisoNET\nbarrier",
        "anisoNET_no_transcript_barrier": "anisoNET\nno barrier",
        "graph_smooth_k6_iter3": "Graph\nk=6",
        "graph_smooth_k12_iter5": "Graph\nk=12",
    }
    return mapping.get(label, str(label))


def clean_synthetic_method_label(label: str) -> str:
    mapping = {
        "nearest": "Near",
        "idw_k8": "IDW",
        "gaussian_sigma3": "Gauss3",
        "graph_smooth_k6_iter3": "Graph6",
        "graph_smooth_k12_iter5": "Graph12",
        "anisoNET_original_barrier": "aNET+",
        "anisoNET_no_transcript_barrier": "aNET-",
    }
    return mapping.get(label, str(label))


def plot_heldout(ax: plt.Axes) -> None:
    df = pd.read_csv(ROOT / "heldout" / "brain_aging_gse193107" / "summary" / "heldout_group_summary.csv")
    keep = ["nearest", "idw_k8", "gaussian_sigma1p5", "gaussian_sigma3", "anisoNET_masked", "anisoNET_gauss07"]
    df = df[df["method_label"].isin(keep)].copy()
    grouped = df.groupby(["target_gene", "method_label"], as_index=False).agg(
        mean=("test_pearson_mean", "mean"),
        sem=("test_pearson_sem", lambda x: np.sqrt(np.sum(np.square(x))) / len(x)),
    )
    order = keep
    x_base = np.arange(len(order))
    width = 0.36
    colors = {"Apoe": "#4C78A8", "Gfap": "#B279A2"}
    for i, gene in enumerate(["Apoe", "Gfap"]):
        sub = grouped[grouped["target_gene"] == gene].set_index("method_label").reindex(order)
        x = x_base + (i - 0.5) * width
        ax.bar(x, sub["mean"], width=width, yerr=sub["sem"], label=gene, color=colors[gene], edgecolor="#222222", linewidth=0.5)
    ax.set_xticks(x_base)
    ax.set_xticklabels([clean_method_label(x) for x in order], rotation=35, ha="right")
    ax.set_ylabel("Held-out Pearson")
    ax.set_ylim(0, 0.86)
    ax.set_title("Generic held-out expression prediction", fontsize=9)
    ax.legend(frameon=False, ncols=2, loc="upper left")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)


def plot_synthetic(ax: plt.Axes, metric: str, ylabel: str, title: str, lower_is_better: bool = False) -> None:
    df = pd.read_csv(SYNTHETIC_ROOT / "synthetic_barrier_summary.csv")
    keep = [
        "nearest",
        "idw_k8",
        "gaussian_sigma3",
        "graph_smooth_k6_iter3",
        "graph_smooth_k12_iter5",
        "anisoNET_original_barrier",
        "anisoNET_no_transcript_barrier",
    ]
    df = df[df["method"].isin(keep)].copy()
    df["split"] = df["train_fraction"].map({0.8: "80/20", 0.2: "20/80"})
    order = keep
    x_base = np.arange(len(order))
    width = 0.36
    colors = {"80/20": "#59A14F", "20/80": "#E15759"}
    for i, split in enumerate(["80/20", "20/80"]):
        sub = df[df["split"] == split].set_index("method").reindex(order)
        x = x_base + (i - 0.5) * width
        sd_col = metric.replace("_mean", "_sd")
        ax.bar(
            x,
            sub[metric],
            width=width,
            yerr=sub[sd_col],
            label=f"train/test {split}",
            color=colors[split],
            edgecolor="#222222",
            linewidth=0.5,
        )
    ax.set_xticks(x_base)
    ax.set_xticklabels([clean_synthetic_method_label(x) for x in order], rotation=45, ha="right", fontsize=6.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    if lower_is_better:
        ax.text(0.98, 0.04, "lower is better", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#555555")


def normalize_for_display(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return np.clip((arr - vmin) / (vmax - vmin), 0, 1)


def plot_synthetic_maps(fig: plt.Figure, spec: gridspec.SubplotSpec) -> None:
    seed_dir = SYNTHETIC_ROOT / "seed0"
    map_specs = [
        ("Truth", "truth.npy"),
        ("anisoNET\nbarrier", "anisoNET_original_barrier.npy"),
        ("anisoNET\nno barrier", "anisoNET_no_transcript_barrier.npy"),
        ("Gaussian\nsigma=3", "gaussian_sigma3.npy"),
        ("IDW\nk=8", "idw_k8.npy"),
    ]
    arrays = [(label, np.asarray(np.load(seed_dir / name), dtype=float)) for label, name in map_specs]
    valid = np.concatenate([arr[np.isfinite(arr)].reshape(-1) for _, arr in arrays])
    vmin, vmax = np.percentile(valid, [1, 99])
    inner = gridspec.GridSpecFromSubplotSpec(1, len(arrays), subplot_spec=spec, wspace=0.04)
    image = None
    for i, (label, arr) in enumerate(arrays):
        ax = fig.add_subplot(inner[0, i])
        image = ax.imshow(normalize_for_display(arr, vmin, vmax), cmap="magma", origin="lower", interpolation="bilinear")
        ax.set_title(label, fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i == 0:
            panel_label(ax, "B", x=-0.18, y=1.08)
            ax.text(-0.18, -0.08, "Representative synthetic field maps", transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="top")
    if image is not None:
        cax = fig.add_axes([0.942, 0.455, 0.008, 0.145])
        cbar = fig.colorbar(image, cax=cax)
        cbar.outline.set_linewidth(0.3)
        cbar.ax.tick_params(labelsize=5, length=1.5, width=0.3)


def plot_paired_delta(ax: plt.Axes) -> None:
    df = pd.read_csv(SYNTHETIC_ROOT / "synthetic_barrier_paired_ablation_stats.csv")
    metric_order = [
        ("grid_pearson_truth", "Truth\nPearson"),
        ("grid_mse_truth", "Truth\nMSE"),
        ("high_to_low_barrier_ratio", "Leakage\nratio"),
    ]
    rows = []
    for train_fraction, split_label in [(0.2, "20/80"), (0.8, "80/20")]:
        for metric, label in metric_order:
            row = df[(df["train_fraction"] == train_fraction) & (df["metric"] == metric)].iloc[0]
            rows.append(
                {
                    "split": split_label,
                    "metric": label,
                    "delta": row["paired_difference_mean"],
                    "sd": row["paired_difference_sd"],
                }
            )
    out = pd.DataFrame(rows)
    x = np.arange(len(metric_order))
    width = 0.35
    colors = {"20/80": "#E15759", "80/20": "#59A14F"}
    for i, split in enumerate(["20/80", "80/20"]):
        sub = out[out["split"] == split].set_index("metric").reindex([label for _, label in metric_order])
        ax.bar(
            x + (i - 0.5) * width,
            sub["delta"],
            width=width,
            yerr=sub["sd"],
            color=colors[split],
            edgecolor="#222222",
            linewidth=0.5,
            label=f"train/test {split}",
        )
    ax.axhline(0, color="#222222", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_order])
    ax.set_ylabel("Barrier - no barrier")
    ax.set_title("Paired transcriptomic-barrier ablation", fontsize=9)
    ax.text(0.02, 0.03, "Pearson: higher is better\nMSE/leakage: lower is better", transform=ax.transAxes, fontsize=6.3, va="bottom", color="#555555")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, loc="upper right", fontsize=6.5)


def assemble_figure3() -> None:
    setup_matplotlib()
    fig = plt.figure(figsize=(7.4, 8.4), constrained_layout=False)
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.12, 0.90, 1.05], hspace=0.50, wspace=0.33)

    ax_a = fig.add_subplot(gs[0, 0:3])
    panel_label(ax_a, "A", x=-0.06, y=1.08)
    plot_heldout(ax_a)

    plot_synthetic_maps(fig, gs[1, 0:3])

    ax_c = fig.add_subplot(gs[2, 0])
    panel_label(ax_c, "C", x=-0.18, y=1.08)
    plot_synthetic(ax_c, "grid_pearson_truth_mean", "Grid-truth Pearson", "Synthetic truth reconstruction")
    ax_c.legend(frameon=False, loc="lower right", fontsize=6.5)

    ax_d = fig.add_subplot(gs[2, 1])
    panel_label(ax_d, "D", x=-0.18, y=1.08)
    plot_synthetic(
        ax_d,
        "high_to_low_barrier_ratio_mean",
        "High/low barrier leakage ratio",
        "Synthetic high-barrier leakage",
        lower_is_better=True,
    )
    ax_d.legend(frameon=False, loc="upper left", fontsize=6.5)

    ax_e = fig.add_subplot(gs[2, 2])
    panel_label(ax_e, "E", x=-0.18, y=1.08)
    plot_paired_delta(ax_e)

    fig.suptitle("Benchmark reframing and barrier-controlled validation", fontsize=12, fontweight="bold", y=0.985)
    save_figure(fig, "Figure3_benchmark_and_barrier_validation_draft")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=["all", "figure2", "figure3"], default="all")
    args = parser.parse_args()
    if args.figure in {"all", "figure2"}:
        assemble_figure2()
    if args.figure in {"all", "figure3"}:
        assemble_figure3()
    print(f"Wrote assembled figure drafts to {OUT_DIR}")


if __name__ == "__main__":
    main()
