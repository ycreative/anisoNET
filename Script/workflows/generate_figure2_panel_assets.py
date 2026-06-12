"""Generate separated Figure 2 panel assets for GSE193107 primary application."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ROOT = Path(os.environ.get("ANISONET_ANALYSIS_ROOT", PROJECT_ROOT / "codexAnalysis"))
OUT_DIR = ROOT / "manuscript_figures" / "Figure2_gse193107_primary_application"
FIGURE_VERSION = os.environ.get("ANISONET_FIGURE2_VERSION", "v20260611_03")
PREFLIGHT_ROOT = ROOT / "preflight" / "brain_aging_gse193107"
PINN_ROOT = ROOT / "pinn" / "brain_aging_gse193107"
SPATIAL_ROOT = ROOT / "processed_visium" / "brain_aging_gse193107"
BATCH_ROOT = ROOT / "batch" / "brain_aging_gse193107"
METRIC_SUMMARY = ROOT / "barrier_field_metrics" / "barrier_field_metrics_summary.csv"
TARGETED_ROOT = ROOT / "targeted_gene_extension"
TARGETED_PREFLIGHT_ROOT = TARGETED_ROOT / "preflight" / "brain_aging_gse193107"
TARGETED_PINN_ROOT = TARGETED_ROOT / "pinn" / "brain_aging_gse193107"
TARGETED_GENE_METRICS = TARGETED_ROOT / "full_metrics_by_dataset_gene.csv"
TARGETED_FULL_METRICS = TARGETED_ROOT / "full_metrics_summary.csv"

REP_SAMPLE = "GSM5773457_Old_mouse_brain_A1-2"
REP_TASK = "Apoe_CNS_Myelin"
PINN_RUN = "fourier_refined_16g_gauss07_batch"
TARGETED_PINN_RUN = "fourier_refined_low_pde_16g_gauss07"
TARGETS = {"Apoe": "Apoe_CNS_Myelin", "Gfap": "Gfap_CNS_Myelin"}
MULTIGENE_TARGETS = ["Apoe", "Gfap", "C1qa", "Trem2", "Tyrobp", "Aif1", "Cst3", "Lpl"]
MULTIGENE_COLORS = {
    "Apoe": "#2A6F97",
    "Gfap": "#A75D67",
    "C1qa": "#6A994E",
    "Trem2": "#8A6BBE",
    "Tyrobp": "#B07D2C",
    "Aif1": "#4D908E",
    "Cst3": "#D17A22",
    "Lpl": "#6C757D",
}

ACTIVE_PANEL_ALIASES = {
    "Fig2A_dataset_design": "Fig2_{version}_A_dataset_design",
    "Fig2B_representative_input_stack": "Fig2_{version}_B_representative_input_stack",
    "Fig2C_source_field_delta_barrier_overlay": "Fig2_{version}_C_source_field_delta_barrier_overlay",
    "Fig2D_brain_aging_gene_section_qc_heatmap": "Fig2_{version}_D_gene_section_qc_heatmap",
    "Fig2E_tissue_support_leakage": "Fig2_{version}_E_tissue_support_leakage",
    "Fig2F_barrier_context_summary": "Fig2_{version}_F_barrier_context_summary",
    "Figure2_rough_assembly": "Figure2_{version}_rough_assembly",
}

LOGICAL_PANEL_ALIASES = {
    "Fig2F_tissue_support_leakage": "Fig2E_tissue_support_leakage",
    "Fig2G_barrier_context_summary": "Fig2F_barrier_context_summary",
}


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
    fig.savefig(pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def write_versioned_aliases() -> None:
    for source_stem, target_template in ACTIVE_PANEL_ALIASES.items():
        target_stem = target_template.format(version=FIGURE_VERSION)
        for ext in ("png", "pdf"):
            source = OUT_DIR / f"{source_stem}.{ext}"
            target = OUT_DIR / f"{target_stem}.{ext}"
            if source.exists():
                shutil.copy2(source, target)


def write_logical_aliases() -> None:
    for source_stem, target_stem in LOGICAL_PANEL_ALIASES.items():
        for ext in ("png", "pdf"):
            source = OUT_DIR / f"{source_stem}.{ext}"
            target = OUT_DIR / f"{target_stem}.{ext}"
            if source.exists():
                shutil.copy2(source, target)


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.08) -> None:
    return None


def sample_label(sample: str) -> str:
    condition = "Young" if "_Young_" in sample else "Old"
    replicate = sample.split("_brain_")[-1]
    gsm = sample.split("_")[0].replace("GSM577", "G")
    return f"{condition} {replicate}\n{gsm}"


def short_section_label(sample: str) -> str:
    condition = "Y" if "_Young_" in sample else "O"
    replicate = sample.split("_brain_")[-1].replace("mouse_", "")
    return f"{condition} {replicate}"


def mask_bbox(mask: np.ndarray, pad: int = 5) -> tuple[int, int, int, int]:
    yy, xx = np.where(mask)
    if yy.size == 0:
        return (0, mask.shape[0], 0, mask.shape[1])
    y0 = max(int(yy.min()) - pad, 0)
    y1 = min(int(yy.max()) + pad + 1, mask.shape[0])
    x0 = max(int(xx.min()) - pad, 0)
    x1 = min(int(xx.max()) + pad + 1, mask.shape[1])
    return (y0, y1, x0, x1)


def crop_with_bbox(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return arr[y0:y1, x0:x1]


def crop_to_mask(arr: np.ndarray, mask: np.ndarray, pad: int = 5) -> np.ndarray:
    return crop_with_bbox(arr, mask_bbox(mask, pad=pad))


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


def smooth_masked_display(panel: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    valid = mask & np.isfinite(panel)
    if not np.any(valid):
        return panel
    values = np.where(valid, panel, 0.0)
    weights = valid.astype(float)
    smooth = gaussian_filter(values, sigma=sigma, mode="nearest")
    norm = gaussian_filter(weights, sigma=sigma, mode="nearest")
    return np.where(mask, smooth / np.maximum(norm, 1e-6), np.nan)


def upsample_display(panel: np.ndarray, min_short_side: int = 720) -> np.ndarray:
    """Upsample display rasters so PDF exports do not embed tiny 80-100 px images."""
    data = np.asarray(panel, dtype=float)
    if data.ndim != 2 or data.size == 0:
        return data
    short_side = min(data.shape)
    if short_side <= 0 or short_side >= min_short_side:
        return data
    scale = min_short_side / short_side
    valid = np.isfinite(data)
    values = np.where(valid, data, 0.0)
    weights = valid.astype(float)
    values_z = zoom(values, scale, order=3, mode="nearest")
    weights_z = zoom(weights, scale, order=1, mode="nearest")
    out = values_z / np.maximum(weights_z, 1e-6)
    return np.where(weights_z > 0.35, out, np.nan)


def show_grid(
    ax: plt.Axes,
    arr: np.ndarray,
    mask: np.ndarray,
    title: str,
    cmap: str,
    *,
    log: bool = False,
    display_smooth_sigma: float | None = None,
) -> None:
    data = normalized_panel(arr, mask, log=log)
    if display_smooth_sigma is not None and display_smooth_sigma > 0:
        data = smooth_masked_display(data, mask, display_smooth_sigma)
    panel = crop_to_mask(data, mask)
    panel = upsample_display(panel)
    ax.imshow(panel, cmap=cmap, interpolation="bicubic", origin="lower")
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
        ("Targets", "8 brain-aging genes"),
        ("Barrier prior", "CNS-myelin: Mbp, Plp1, Mobp"),
        ("Model profile", "scalar PINN + tissue mask"),
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
    show_grid(axes[1], source, mask, "Source\nApoe", "magma", display_smooth_sigma=5.0)
    show_grid(axes[2], barrier, mask, "Barrier\nCNS myelin", "YlOrBr", display_smooth_sigma=5.0)
    show_grid(axes[3], resistance, mask, "Resistance\nR=1/D", "viridis", log=True, display_smooth_sigma=5.0)
    show_grid(axes[4], diffusion, mask, "Diffusion\nD(x,y)", "cividis", display_smooth_sigma=5.0)
    show_grid(axes[5], field, mask, "Postprocessed\nfield", "plasma")
    fig.suptitle("Representative old A1 input and output stack", fontsize=9, fontweight="bold", y=1.03)
    save_panel(fig, "Fig2B_representative_input_stack")


def display_cropped_map(
    arr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    log: bool = False,
    sigma: float | None = None,
    min_short_side: int = 720,
) -> np.ndarray:
    panel = normalized_panel(arr, mask, log=log)
    if sigma is not None and sigma > 0:
        panel = smooth_masked_display(panel, mask, sigma)
    panel = upsample_display(crop_with_bbox(panel, bbox), min_short_side=min_short_side)
    return np.clip(panel, 0, 1)


def fig2c_source_field_delta_barrier_overlay() -> None:
    genes = ["Apoe", "Gfap"]
    fig = plt.figure(figsize=(7.1, 3.8))
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.07, hspace=0.18)

    for row, gene in enumerate(genes):
        task = TARGETS[gene]
        preflight = PREFLIGHT_ROOT / REP_SAMPLE / task
        pinn = PINN_ROOT / REP_SAMPLE / task / PINN_RUN
        mask = np.load(preflight / "tissue_mask.npy") > 0
        bbox = mask_bbox(mask)
        source = np.load(preflight / "source_grid.npy")
        barrier = np.load(preflight / "barrier_grid.npy")
        field = np.load(pinn / "pinn_grid_prediction_postprocessed.npy")

        source_panel = display_cropped_map(source, mask, bbox, sigma=5.0)
        field_panel = display_cropped_map(field, mask, bbox, sigma=1.0)
        barrier_panel = display_cropped_map(barrier, mask, bbox, sigma=5.0)
        source_norm = normalized_panel(source, mask)
        field_norm = normalized_panel(field, mask)
        delta = smooth_masked_display(field_norm, mask, 1.0) - smooth_masked_display(source_norm, mask, 5.0)
        delta = np.where(mask, delta, np.nan)
        delta_panel = upsample_display(crop_with_bbox(delta, bbox))
        delta_valid = np.abs(delta_panel[np.isfinite(delta_panel)])
        delta_limit = float(np.nanpercentile(delta_valid, 98)) if delta_valid.size else 1.0
        if not np.isfinite(delta_limit) or delta_limit <= 0:
            delta_limit = 1.0

        panels = [
            ("Measured source\nS(x,y)", source_panel, "magma", None),
            ("Inferred field\nC(x,y)", field_panel, "plasma", None),
            ("Field - source\n(display scale)", delta_panel, "coolwarm", delta_limit),
            ("Field with high\nbarrier contour", field_panel, "plasma", None),
        ]
        for col, (title, panel, cmap, limit) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            if row == 0 and col == 0:
                panel_label(ax, "C", x=-0.24, y=1.08)
            if limit is None:
                ax.imshow(panel, origin="lower", cmap=cmap, vmin=0, vmax=1, interpolation="bicubic")
            else:
                ax.imshow(
                    panel,
                    origin="lower",
                    cmap=cmap,
                    vmin=-limit,
                    vmax=limit,
                    interpolation="bicubic",
                )
            if col == 3:
                valid = barrier_panel[np.isfinite(barrier_panel)]
                if valid.size:
                    threshold = float(np.nanpercentile(valid, 80))
                    ax.contour(barrier_panel, levels=[threshold], colors=["#00A6C8"], linewidths=1.0)
                ax.text(
                    0.03,
                    0.04,
                    "cyan: high CNS-myelin barrier",
                    transform=ax.transAxes,
                    fontsize=5.6,
                    color="#1f4d57",
                    ha="left",
                    va="bottom",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
                )
            if row == 0:
                ax.set_title(title, fontsize=7.2, pad=2)
            if col == 0:
                ax.text(
                    -0.10,
                    0.50,
                    gene,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=7.3,
                    fontweight="bold",
                    color=MULTIGENE_COLORS.get(gene, "#333333"),
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle("Representative source-to-field change and barrier boundary context", fontsize=9, fontweight="bold", y=1.02)
    save_panel(fig, "Fig2C_source_field_delta_barrier_overlay")


def fig2d_gene_section_qc_heatmap() -> None:
    df = pd.read_csv(TARGETED_FULL_METRICS)
    df = df[df["dataset"] == "brain_aging_gse193107"].copy()
    df["target_gene"] = pd.Categorical(df["target_gene"], categories=MULTIGENE_TARGETS, ordered=True)
    samples = sorted(df["sample"].dropna().unique(), key=lambda name: (0 if "_Young_" in name else 1, name))
    metrics = [
        ("spot_pearson_source", "Fitted-source\nPearson", "viridis", (0.55, 0.90), None),
        ("roughness_grad_p95", "Field roughness\np95 gradient", "magma_r", None, None),
        ("high_to_low_barrier_prediction_ratio", "High/low barrier\nfield ratio", "coolwarm", (0.7, 2.1), 1.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), gridspec_kw={"wspace": 0.26})
    panel_label(axes[0], "D", x=-0.14, y=1.10)
    for ax, (value_col, title, cmap, fixed_range, center) in zip(axes, metrics):
        table = (
            df.pivot_table(index="target_gene", columns="sample", values=value_col, observed=False)
            .reindex(index=MULTIGENE_TARGETS, columns=samples)
            .astype(float)
        )
        values = table.to_numpy()
        if fixed_range is None:
            finite = values[np.isfinite(values)]
            vmin, vmax = np.nanpercentile(finite, [3, 97]) if finite.size else (0, 1)
        else:
            vmin, vmax = fixed_range
        if center is not None:
            norm_obj = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        else:
            norm_obj = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm_obj)
        ax.set_title(title, fontsize=7.3, pad=3)
        ax.set_xticks(np.arange(len(samples)))
        ax.set_xticklabels([short_section_label(s) for s in samples], rotation=45, ha="right", fontsize=5.7)
        ax.set_yticks(np.arange(len(MULTIGENE_TARGETS)))
        ax.set_yticklabels(MULTIGENE_TARGETS if ax is axes[0] else [], fontsize=6.3)
        ax.set_xticks(np.arange(-0.5, len(samples), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(MULTIGENE_TARGETS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
        cbar.outline.set_linewidth(0.3)
        cbar.ax.tick_params(labelsize=5.2, length=1.4, width=0.3)
    axes[2].text(
        0.98,
        -0.30,
        "Pearson is fitted-source QC, not held-out prediction.",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color="#6b1d1d",
    )
    fig.suptitle("Brain-aging targeted extension QC across 8 genes x 8 sections", fontsize=9, fontweight="bold", y=1.03)
    save_panel(fig, "Fig2D_brain_aging_gene_section_qc_heatmap")


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
        fields.append((sample, upsample_display(crop_to_mask(np.where(mask, field, np.nan), mask))))
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


def targeted_paths(sample: str, gene: str) -> tuple[Path, Path]:
    task = f"{gene}_CNS_Myelin"
    preflight = TARGETED_PREFLIGHT_ROOT / sample / task / "brightness"
    pinn = TARGETED_PINN_ROOT / sample / task / TARGETED_PINN_RUN
    return preflight, pinn


def fig2c_multigene_old_a1_fields() -> None:
    fig = plt.figure(figsize=(5.85, 5.45))
    gs = gridspec.GridSpec(4, 4, figure=fig, wspace=0.06, hspace=0.30)
    for idx, gene in enumerate(MULTIGENE_TARGETS):
        preflight, pinn = targeted_paths(REP_SAMPLE, gene)
        mask = np.load(preflight / "tissue_mask.npy") > 0
        source = np.load(preflight / "source_grid.npy")
        field = np.load(pinn / "pinn_grid_prediction_postprocessed.npy")

        row = idx // 2
        col = (idx % 2) * 2
        ax_source = fig.add_subplot(gs[row, col])
        ax_field = fig.add_subplot(gs[row, col + 1])
        if idx == 0:
            panel_label(ax_source, "C", x=-0.18, y=1.12)
        show_grid(ax_source, source, mask, f"{gene}\nsource", "magma", display_smooth_sigma=5.0)
        show_grid(ax_field, field, mask, "field", "plasma", display_smooth_sigma=1.0)
    fig.suptitle("Representative old A1 multi-gene source and fitted fields", fontsize=9, fontweight="bold", y=1.01)
    save_panel(fig, "Fig2C_brain_aging_multigene_oldA1_fields")


def fig2d_multigene_8section_mosaic() -> None:
    samples = sorted([path.name for path in TARGETED_PINN_ROOT.iterdir() if path.is_dir()])
    fig = plt.figure(figsize=(7.6, 7.15))
    gs = gridspec.GridSpec(
        len(MULTIGENE_TARGETS),
        len(samples),
        figure=fig,
        wspace=0.04,
        hspace=0.08,
    )
    for row, gene in enumerate(MULTIGENE_TARGETS):
        tissue_values = []
        cropped_fields: list[tuple[str, np.ndarray]] = []
        for sample in samples:
            preflight, pinn = targeted_paths(sample, gene)
            mask = np.load(preflight / "tissue_mask.npy") > 0
            field = np.load(pinn / "pinn_grid_prediction_postprocessed.npy")
            tissue_values.append(field[mask].reshape(-1))
            display_field = smooth_masked_display(field, mask, sigma=1.0)
            cropped_fields.append((sample, crop_to_mask(display_field, mask)))
        vmax = float(np.nanpercentile(np.concatenate(tissue_values), 99.5))
        for col, (sample, field) in enumerate(cropped_fields):
            ax = fig.add_subplot(gs[row, col])
            if row == 0 and col == 0:
                panel_label(ax, "D", x=-0.55, y=1.20)
            ax.imshow(upsample_display(field, min_short_side=420), origin="lower", cmap="magma", vmin=0, vmax=vmax)
            if row == 0:
                ax.set_title(sample_label(sample).split("\n")[0], fontsize=5.4, pad=1.2)
            if col == 0:
                ax.text(
                    -0.08,
                    0.50,
                    gene,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=6.4,
                    fontweight="bold",
                    color=MULTIGENE_COLORS.get(gene, "#333333"),
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle("Eight brain-aging genes across eight GSE193107 sections", fontsize=9, fontweight="bold", y=0.995)
    save_panel(fig, "Fig2D_brain_aging_multigene_8section_fields")


def fig2e_multigene_source_fidelity_roughness() -> None:
    df = pd.read_csv(TARGETED_GENE_METRICS)
    df = df[df["dataset"] == "brain_aging_gse193107"].copy()
    df["target_gene"] = pd.Categorical(df["target_gene"], categories=MULTIGENE_TARGETS, ordered=True)
    df = df.sort_values("target_gene")
    colors = [MULTIGENE_COLORS.get(str(gene), "#666666") for gene in df["target_gene"]]
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.35), gridspec_kw={"wspace": 0.34})
    panel_label(axes[0], "E", x=-0.14, y=1.10)
    axes[0].bar(x, df["spot_pearson_source_mean"], color=colors, edgecolor="#222222", linewidth=0.45)
    axes[0].vlines(
        x,
        df["spot_pearson_source_min"],
        df["spot_pearson_source_max"],
        color="#222222",
        linewidth=0.65,
    )
    axes[0].set_ylim(0.55, 0.90)
    axes[0].set_ylabel("Source Pearson")
    axes[0].set_title("Fitted-source fidelity")

    axes[1].bar(x, df["roughness_grad_p95_mean"], color=colors, edgecolor="#222222", linewidth=0.45)
    axes[1].set_ylabel("Gradient p95")
    axes[1].set_title("Field roughness")

    axes[2].bar(x, df["high_to_low_barrier_prediction_ratio_mean"], color=colors, edgecolor="#222222", linewidth=0.45)
    axes[2].axhline(1.0, color="#555555", linewidth=0.65, linestyle="--")
    axes[2].set_ylabel("High/low barrier ratio")
    axes[2].set_title("Barrier context")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in df["target_gene"]], rotation=45, ha="right", fontsize=6.2)
        ax.grid(axis="y", color="#dddddd", linewidth=0.45)
    fig.suptitle("Brain-aging targeted extension across 8 genes x 8 sections", fontsize=9, fontweight="bold", y=1.04)
    save_panel(fig, "Fig2E_brain_aging_multigene_source_fidelity_and_roughness")


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
        & (summary["field_type"] == "raw")
        & (summary["metric"] == "tissue_support_leakage")
    ].copy()
    leakage["label"] = leakage["target"] + "\n" + leakage["condition"]
    order = ["Apoe\nYoung", "Apoe\nOld", "Gfap\nYoung", "Gfap\nOld"]
    x_base = np.arange(len(order))
    sub = leakage.set_index("label").reindex(order)

    preflight = PREFLIGHT_ROOT / REP_SAMPLE / REP_TASK
    pinn = PINN_ROOT / REP_SAMPLE / REP_TASK / PINN_RUN
    mask = np.load(preflight / "tissue_mask.npy") > 0
    bbox = mask_bbox(mask, pad=8)
    raw = np.load(pinn / "pinn_grid_prediction_raw.npy")
    postprocessed = np.load(pinn / "pinn_grid_prediction_postprocessed.npy")
    valid = np.isfinite(raw)
    lo, hi = np.nanpercentile(raw[valid], [2, 98]) if np.any(valid) else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1e-6
    raw_panel = np.clip((raw - lo) / (hi - lo), 0, 1)
    post_panel = np.clip((postprocessed - lo) / (hi - lo), 0, 1)
    post_panel = np.where(mask, post_panel, np.nan)

    fig = plt.figure(figsize=(5.5, 3.05))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.20, 0.95], wspace=0.18, hspace=0.38)

    map_specs = [
        ("Raw PINN\nfield", upsample_display(crop_with_bbox(raw_panel, bbox), min_short_side=520), "plasma", "bicubic"),
        ("Tissue\nsupport", crop_with_bbox(mask.astype(float), bbox), "Greys", "nearest"),
        ("Masked\nfield", upsample_display(crop_with_bbox(post_panel, bbox), min_short_side=520), "plasma", "bicubic"),
    ]
    for idx, (title, panel, cmap, interpolation) in enumerate(map_specs):
        ax = fig.add_subplot(gs[0, idx])
        if idx == 0:
            panel_label(ax, "E", x=-0.18, y=1.10)
        ax.imshow(panel, origin="lower", cmap=cmap, vmin=0, vmax=1, interpolation=interpolation)
        ax.set_title(title, fontsize=7.3, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax_metric = fig.add_subplot(gs[1, :])
    y = np.arange(len(order))[::-1]
    values = sub["value_mean"].to_numpy(dtype=float)
    sem = sub["value_sem"].to_numpy(dtype=float)
    colors = ["#D17A22", "#D17A22", "#B279A2", "#B279A2"]
    for yi, value, err, color in zip(y, values, sem, colors):
        ax_metric.errorbar(
            value,
            yi,
            xerr=err,
            fmt="o",
            color=color,
            ecolor="#222222",
            elinewidth=0.8,
            capsize=2.5,
            markersize=4.8,
            markeredgecolor="#222222",
            markeredgewidth=0.35,
            zorder=3,
        )
    xmin = max(0.0, float(np.nanmin(values - sem)) - 0.02)
    xmax = float(np.nanmax(values + sem)) + 0.03
    ax_metric.set_xlim(xmin, xmax)
    ax_metric.set_yticks(y)
    ax_metric.set_yticklabels(order, fontsize=6.4)
    ax_metric.set_xlabel("Raw off-tissue field mass")
    ax_metric.set_title("Background field mass (mean +/- SEM)", fontsize=7.3, pad=2)
    ax_metric.grid(axis="x", color="#dddddd", linewidth=0.45)
    ax_metric.tick_params(axis="x", labelsize=6.0)
    ax_metric.text(
        0.99,
        -0.52,
        "Example maps: Apoe old A1; masking removes off-tissue field support.",
        transform=ax_metric.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color="#555555",
    )
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
    fig, ax = plt.subplots(figsize=(3.75, 2.85))
    panel_label(ax, "F", x=-0.16, y=1.10)
    y = np.arange(len(order))[::-1]
    values = sub["value_mean"].to_numpy(dtype=float)
    sem = sub["value_sem"].to_numpy(dtype=float)
    ax.axvline(1.0, color="#555555", linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
    for yi, value, err, color in zip(y, values, sem, colors):
        ax.plot([1.0, value], [yi, yi], color=color, linewidth=1.6, alpha=0.8, zorder=2)
        ax.errorbar(
            value,
            yi,
            xerr=err,
            fmt="o",
            color=color,
            ecolor="#222222",
            elinewidth=0.85,
            capsize=2.8,
            markersize=5.2,
            markeredgecolor="#222222",
            markeredgewidth=0.35,
            zorder=3,
        )
    xmin = min(0.85, float(np.nanmin(values - sem)) - 0.05)
    xmax = max(1.75, float(np.nanmax(values + sem)) + 0.08)
    ax.set_xlim(xmin, xmax)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=6.8)
    ax.set_xlabel("High/low barrier field ratio")
    ax.set_title("Barrier-context field summary", fontsize=8.0, pad=3)
    ax.grid(axis="x", color="#dddddd", linewidth=0.45)
    ax.tick_params(axis="x", labelsize=6.2)
    ax.text(
        0.02,
        -0.34,
        "Dashed line: equal high/low barrier field. Descriptive for real tissue.",
        transform=ax.transAxes,
        fontsize=5.8,
        color="#555555",
        ha="left",
        va="top",
    )
    save_panel(fig, "Fig2G_barrier_context_summary")


def rough_assembly() -> None:
    panel_files = [
        "Fig2A_dataset_design.png",
        "Fig2B_representative_input_stack.png",
        "Fig2C_source_field_delta_barrier_overlay.png",
        "Fig2D_brain_aging_gene_section_qc_heatmap.png",
        "Fig2E_tissue_support_leakage.png",
        "Fig2F_barrier_context_summary.png",
    ]
    fig = plt.figure(figsize=(10.2, 10.8))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[0.76, 1.75, 1.15, 0.82], hspace=0.20, wspace=0.12)
    layout = [
        (panel_files[0], gs[0, 0]),
        (panel_files[1], gs[0, 1]),
        (panel_files[2], gs[1, :]),
        (panel_files[3], gs[2, :]),
        (panel_files[4], gs[3, 0]),
        (panel_files[5], gs[3, 1]),
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
    fig2c_source_field_delta_barrier_overlay()
    fig2d_gene_section_qc_heatmap()
    fig2_field_montage("Apoe", "C")
    fig2_field_montage("Gfap", "D")
    fig2c_multigene_old_a1_fields()
    fig2d_multigene_8section_mosaic()
    fig2e_source_fidelity_roughness()
    fig2e_multigene_source_fidelity_roughness()
    fig2f_tissue_support_metrics()
    fig2g_barrier_context_summary()
    write_logical_aliases()
    rough_assembly()
    write_versioned_aliases()
    print(f"Wrote Figure 2 panel assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
