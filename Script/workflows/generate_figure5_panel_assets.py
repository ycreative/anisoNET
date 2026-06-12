"""Generate separated Figure 5 panel assets for cross-tissue evidence boundaries."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from scipy import sparse
from PIL import Image


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ROOT = Path(os.environ.get("ANISONET_ANALYSIS_ROOT", PROJECT_ROOT / "codexAnalysis"))
OUT_DIR = ROOT / "manuscript_figures" / "Figure5_cross_tissue_boundary"
EXPORT_DPI = 600

PROCESSED = ROOT / "processed_visium"
KIDNEY_PROCESSED = PROCESSED / "mouse_kidney_10x" / "V1_Mouse_Kidney"
LIVER_PROCESSED = PROCESSED / "mouse_liver_apap_gse280515" / "GSM8599603_liver_APAP48h_rep1_ABU007"
SAGITTAL_PROCESSED = PROCESSED / "mouse_brain_sagittal_10x" / "V1_Mouse_Brain_Sagittal_Anterior_Section1"

EVIDENCE = ROOT / "cross_tissue" / "evidence_summary" / "cross_tissue_evidence_summary_table.csv"
KIDNEY_MAIN = ROOT / "barrier_split_anisonet" / "mouse_kidney_10x" / "V1_Mouse_Kidney" / "evidence_boundary_diagnostics" / "kidney_main_task_evidence_boundary.csv"
KIDNEY_MARKER = ROOT / "barrier_split_anisonet" / "mouse_kidney_10x" / "V1_Mouse_Kidney" / "evidence_boundary_diagnostics" / "kidney_marker_screen_evidence_boundary.csv"
MARKER_RANK = ROOT / "cross_tissue" / "mouse_kidney_10x" / "V1_Mouse_Kidney" / "marker_task_screen" / "kidney_marker_task_screen_top_recommendations.csv"
TARGETED_GENE = ROOT / "targeted_gene_extension" / "full_metrics_by_dataset_gene.csv"

PAPER_COLORS = {
    "teal": "#4C9A91",
    "blue": "#5E81AC",
    "terracotta": "#C36F5B",
    "lavender": "#8C7BB7",
    "ochre": "#C7A35A",
    "sage": "#7BA05B",
    "slate": "#7E8796",
    "dark": "#2F3A45",
}

ROLE_COLORS = {
    "method-development": PAPER_COLORS["teal"],
    "supplementary": PAPER_COLORS["sage"],
    "supplementary positive-leaning": PAPER_COLORS["sage"],
    "supplementary mixed": PAPER_COLORS["ochre"],
    "method diagnostic": PAPER_COLORS["slate"],
    "claim-boundary": PAPER_COLORS["terracotta"],
    "claim-boundary / negative control": PAPER_COLORS["slate"],
}


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.4,
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_panel(fig: plt.Figure, stem: str) -> None:
    ensure_dir(OUT_DIR)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.10) -> None:
    return None


def clean_dataset(name: str) -> str:
    return {
        "10x mouse kidney": "Kidney\n10x",
        "mouse liver APAP GSE280515": "Liver\nAPAP",
        "10x mouse brain sagittal": "Brain\nsagittal",
    }.get(name, name)


def clean_task(task: str) -> str:
    return (
        str(task)
        .replace("_", " ")
        .replace("healthy brain portability", "healthy brain\nportability")
        .replace("annotation boundary", "annotation\nboundary")
        .replace("marker screen followup", "marker\nscreen")
        .replace("primary kidney task", "primary\nkidney")
    )


def normalize_values(values: np.ndarray) -> np.ndarray:
    data = np.log1p(np.asarray(values, dtype=float))
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data)
    lo, hi = np.nanpercentile(finite, [5, 99])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((data - lo) / (hi - lo), 0, 1)


def spot_triangulation(spatial: dict) -> mtri.Triangulation | None:
    x = np.asarray(spatial["x"], dtype=float)
    y = np.asarray(spatial["y"], dtype=float)
    if x.size < 4:
        return None
    triangulation = mtri.Triangulation(x, y)
    triangles = triangulation.triangles
    pts = np.column_stack([x, y])
    edge_lengths = []
    for pair in [(0, 1), (1, 2), (2, 0)]:
        edge = pts[triangles[:, pair[0]]] - pts[triangles[:, pair[1]]]
        edge_lengths.append(np.sqrt(np.sum(edge * edge, axis=1)))
    max_edge = np.max(np.vstack(edge_lengths), axis=0)
    limit = np.nanpercentile(max_edge, 92) * 1.35
    triangulation.set_mask(max_edge > limit)
    return triangulation


def overlay_marker_boundary(
    ax: plt.Axes,
    spatial: dict,
    values: np.ndarray,
    color: tuple[float, float, float] | str,
    *,
    level: float | None = None,
    linewidth: float = 0.85,
) -> None:
    triangulation = spot_triangulation(spatial)
    if triangulation is None:
        return
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0 or np.nanmax(finite) <= np.nanmin(finite):
        return
    contour_level = float(level if level is not None else max(0.62, np.nanpercentile(finite, 88)))
    contour_level = min(max(contour_level, float(np.nanmin(finite)) + 1e-6), float(np.nanmax(finite)) - 1e-6)
    try:
        ax.tricontour(triangulation, vals, levels=[contour_level], colors=[color], linewidths=linewidth, alpha=0.95)
    except ValueError:
        return


def load_spatial_dataset(dataset_root: Path, genes: list[str]) -> dict:
    h5_path = dataset_root / "filtered_feature_bc_matrix.h5"
    with h5py.File(h5_path, "r") as handle:
        names = [x.decode("utf-8") for x in handle["matrix/features/name"][:]]
        barcodes = [x.decode("utf-8") for x in handle["matrix/barcodes"][:]]
        mat = sparse.csc_matrix(
            (
                handle["matrix/data"][:],
                handle["matrix/indices"][:],
                handle["matrix/indptr"][:],
            ),
            shape=tuple(handle["matrix/shape"][:]),
        )
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        expr_all = {}
        for gene in genes:
            if gene not in name_to_idx:
                expr_all[gene] = np.zeros(len(barcodes), dtype=float)
            else:
                expr_all[gene] = np.asarray(mat[name_to_idx[gene], :].todense()).reshape(-1)

    pos = pd.read_csv(
        dataset_root / "spatial" / "tissue_positions_list.csv",
        header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"],
    )
    pos = pos[pos["in_tissue"].eq(1)].copy()
    barcode_to_col = {barcode: idx for idx, barcode in enumerate(barcodes)}
    pos = pos[pos["barcode"].isin(barcode_to_col)].copy()
    order = [barcode_to_col[x] for x in pos["barcode"]]

    with (dataset_root / "spatial" / "scalefactors_json.json").open("r", encoding="utf-8") as handle:
        scale = float(json.load(handle)["tissue_hires_scalef"])
    image = Image.open(dataset_root / "spatial" / "tissue_hires_image.png").convert("RGB")
    image_arr = np.asarray(image)
    expr = {gene: values[order] for gene, values in expr_all.items()}
    return {
        "image": image_arr,
        "x": pos["pxl_col"].to_numpy(dtype=float) * scale,
        "y": pos["pxl_row"].to_numpy(dtype=float) * scale,
        "expr": expr,
    }


def show_he(ax: plt.Axes, spatial: dict, title: str) -> None:
    ax.imshow(spatial["image"])
    ax.scatter(spatial["x"], spatial["y"], s=2.0, facecolor="none", edgecolor="#3f3f3f", linewidth=0.12, alpha=0.28)
    ax.set_title(title, fontsize=7.2, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def show_tissue_thumbnail(ax: plt.Axes, dataset_root: Path, title: str) -> None:
    image = Image.open(dataset_root / "spatial" / "tissue_hires_image.png").convert("RGB")
    image.thumbnail((700, 700), Image.Resampling.LANCZOS)
    ax.imshow(np.asarray(image))
    ax.set_title(title, fontsize=7.0, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def show_gene(ax: plt.Axes, spatial: dict, gene: str, title: str, cmap: str) -> None:
    gray = np.asarray(Image.fromarray(spatial["image"]).convert("L"))
    ax.imshow(gray, cmap="gray", alpha=0.40)
    vals = normalize_values(spatial["expr"][gene])
    ax.scatter(spatial["x"], spatial["y"], s=7.0, c=vals, cmap=cmap, vmin=0, vmax=1, edgecolor="none", alpha=0.92)
    ax.set_title(title, fontsize=7.2, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def show_score(ax: plt.Axes, spatial: dict, genes: list[str], title: str, cmap: str) -> None:
    score = np.mean([normalize_values(spatial["expr"][gene]) for gene in genes], axis=0)
    gray = np.asarray(Image.fromarray(spatial["image"]).convert("L"))
    ax.imshow(gray, cmap="gray", alpha=0.40)
    ax.scatter(spatial["x"], spatial["y"], s=7.0, c=score, cmap=cmap, vmin=0, vmax=1, edgecolor="none", alpha=0.92)
    ax.set_title(title, fontsize=7.2, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def show_rgb_composite(ax: plt.Axes, spatial: dict, gene_colors: dict[str, tuple[float, float, float]], title: str) -> None:
    gray = np.asarray(Image.fromarray(spatial["image"]).convert("L"))
    ax.imshow(gray, cmap="gray", alpha=0.35)
    weights = []
    colors = []
    for gene, color in gene_colors.items():
        weights.append(normalize_values(spatial["expr"][gene]))
        colors.append(np.asarray(color, dtype=float))
    weight_mat = np.vstack(weights)
    color_mat = np.vstack(colors)
    raw_rgb = weight_mat.T @ color_mat
    denom = np.maximum(weight_mat.sum(axis=0), 1.0)[:, None]
    intensity = np.clip(weight_mat.max(axis=0), 0, 1)
    mixed = np.clip(raw_rgb / denom, 0, 1)
    rgb = np.clip(0.80 * (1 - intensity[:, None]) + mixed * (0.25 + 0.90 * intensity[:, None]), 0, 1)
    rgba = np.column_stack([rgb, 0.28 + 0.68 * intensity])
    ax.scatter(spatial["x"], spatial["y"], s=8.5, c=rgba, edgecolor="white", linewidth=0.08)
    for gene, color in gene_colors.items():
        overlay_marker_boundary(ax, spatial, normalize_values(spatial["expr"][gene]), color, linewidth=0.75)
    ax.set_title(title, fontsize=7.2, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    y = 0.04
    for gene, color in gene_colors.items():
        ax.text(
            0.03,
            y,
            gene,
            transform=ax.transAxes,
            fontsize=5.8,
            color=color,
            fontweight="bold",
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 0.5},
        )
        y += 0.08


def fig5a_kidney_spatial_context() -> None:
    spatial = load_spatial_dataset(KIDNEY_PROCESSED, ["Slc34a1", "Umod", "Slc12a3", "Slc12a1"])
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 4.3))
    flat = axes.ravel()
    panel_label(flat[0], "A", x=-0.16, y=1.08)
    show_he(flat[0], spatial, "Kidney H&E + Visium")
    show_rgb_composite(
        flat[1],
        spatial,
        {"Slc34a1": (0.90, 0.30, 0.12), "Umod": (0.10, 0.52, 0.95), "Slc12a3": (0.10, 0.68, 0.36)},
        "Compartment marker RGB",
    )
    show_gene(flat[2], spatial, "Slc34a1", "Slc34a1\nproximal tubule", "YlOrRd")
    show_gene(flat[3], spatial, "Umod", "Umod\nTAL/medulla", "Blues")
    show_gene(flat[4], spatial, "Slc12a3", "Slc12a3\nDCT/CNT", "Greens")
    show_score(flat[5], spatial, ["Umod", "Slc12a1"], "TAL barrier score\nUmod + Slc12a1", "PuBuGn")
    fig.suptitle("Kidney task context: spatially separated source and barrier compartments", fontsize=9, fontweight="bold", y=1.02)
    save_panel(fig, "Fig5A_kidney_spatial_gene_context")


def fig5c_liver_spatial_context() -> None:
    spatial = load_spatial_dataset(LIVER_PROCESSED, ["Ass1", "Cps1", "Cyp2f2", "Lyz2", "Spp1"])
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 4.3))
    flat = axes.ravel()
    panel_label(flat[0], "C", x=-0.16, y=1.08)
    show_he(flat[0], spatial, "APAP liver H&E + Visium")
    show_rgb_composite(
        flat[1],
        spatial,
        {"Ass1": (0.83, 0.32, 0.45), "Cps1": (0.10, 0.65, 0.42), "Cyp2f2": (0.35, 0.38, 0.92)},
        "Annotation-boundary RGB",
    )
    show_gene(flat[2], spatial, "Ass1", "Ass1", "rocket" if "rocket" in plt.colormaps() else "magma")
    show_gene(flat[3], spatial, "Cps1", "Cps1", "Greens")
    show_gene(flat[4], spatial, "Cyp2f2", "Cyp2f2", "Purples")
    show_score(flat[5], spatial, ["Lyz2", "Spp1"], "APAP injury/source\nLyz2 + Spp1", "OrRd")
    fig.suptitle("Liver APAP transfer: annotation-boundary candidates are spatially visible but small-effect", fontsize=9, fontweight="bold", y=1.02)
    save_panel(fig, "Fig5C_liver_spatial_annotation_context")


def fig5d_sagittal_spatial_context() -> None:
    spatial = load_spatial_dataset(SAGITTAL_PROCESSED, ["Apoe", "Gfap", "Mbp", "Plp1"])
    fig, axes = plt.subplots(1, 5, figsize=(8.8, 2.4))
    panel_label(axes[0], "D", x=-0.20, y=1.10)
    show_he(axes[0], spatial, "Sagittal brain H&E")
    show_rgb_composite(
        axes[1],
        spatial,
        {"Apoe": (0.88, 0.28, 0.20), "Gfap": (0.55, 0.25, 0.80), "Mbp": (0.10, 0.55, 0.95)},
        "Healthy-control RGB",
    )
    show_gene(axes[2], spatial, "Apoe", "Apoe", "YlOrRd")
    show_gene(axes[3], spatial, "Gfap", "Gfap", "Purples")
    show_score(axes[4], spatial, ["Mbp", "Plp1"], "Myelin score\nMbp + Plp1", "Blues")
    fig.suptitle("Sagittal healthy brain negative-control context", fontsize=9, fontweight="bold", y=1.05)
    save_panel(fig, "Fig5D_sagittal_spatial_negative_control_context")


def fig5a_dataset_task_matrix() -> None:
    df = pd.read_csv(EVIDENCE)
    counts = df.groupby(["dataset", "evidence_block"]).size().reset_index(name="n")
    datasets = list(dict.fromkeys(df["dataset"]))
    blocks = list(dict.fromkeys(df["evidence_block"]))
    mat = np.zeros((len(datasets), len(blocks)))
    for _, row in counts.iterrows():
        mat[datasets.index(row["dataset"]), blocks.index(row["evidence_block"])] = row["n"]
    fig = plt.figure(figsize=(8.6, 3.05))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.95, 0.95, 0.95, 3.15], wspace=0.22)
    thumb_roots = [KIDNEY_PROCESSED, LIVER_PROCESSED, SAGITTAL_PROCESSED]
    thumb_titles = ["Kidney 10x", "Liver APAP", "Brain sagittal"]
    for i, (root, title) in enumerate(zip(thumb_roots, thumb_titles)):
        ax_thumb = fig.add_subplot(gs[0, i])
        if i == 0:
            panel_label(ax_thumb, "E", x=-0.22, y=1.08)
        show_tissue_thumbnail(ax_thumb, root, title)
    ax = fig.add_subplot(gs[0, 3])
    max_n = max(1, mat.max())
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            n = mat[i, j]
            if n <= 0:
                ax.scatter(j, i, s=22, facecolor="white", edgecolor="#d0d0d0", linewidth=0.5)
            else:
                ax.scatter(j, i, s=65 + 105 * n / max_n, color=plt.cm.YlGnBu(0.30 + 0.60 * n / max_n), edgecolor="#333333", linewidth=0.35)
                ax.text(j, i, f"{int(n)}", ha="center", va="center", fontsize=6.4, color="#1f2933")
    ax.set_xlim(-0.6, len(blocks) - 0.4)
    ax.set_ylim(len(datasets) - 0.55, -0.55)
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels([clean_dataset(x) for x in datasets], fontsize=7.0)
    ax.set_xticks(np.arange(len(blocks)))
    ax.set_xticklabels([clean_task(x) for x in blocks], rotation=35, ha="right", fontsize=6.4)
    ax.grid(color="#e5e5e5", linewidth=0.45)
    ax.set_title("Evidence coverage matrix", loc="left", fontsize=8.4)
    ax.text(0.02, -0.22, "Circle area encodes evidence rows", transform=ax.transAxes, fontsize=6.4, color="#5b6470")
    save_panel(fig, "Fig5E_cross_tissue_task_matrix")


def fig5b_kidney_main_boundary() -> None:
    df = pd.read_csv(KIDNEY_MAIN)
    methods = ["resistance_idw_pearson", "line_prior_pearson", "grid_geodesic_pearson", "best_hybrid_pearson"]
    labels = ["Resistance IDW", "Line prior", "Grid geodesic", "Best hybrid"]
    colors = [PAPER_COLORS["slate"], PAPER_COLORS["sage"], PAPER_COLORS["ochre"], PAPER_COLORS["teal"]]
    spatial = load_spatial_dataset(KIDNEY_PROCESSED, ["Slc34a1", "Umod", "Slc12a3", "Slc12a1"])
    fig = plt.figure(figsize=(8.4, 2.75))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1.55, 1.55], wspace=0.28)
    ax_he = fig.add_subplot(gs[0, 0])
    ax_slc = fig.add_subplot(gs[0, 1])
    ax_umod = fig.add_subplot(gs[0, 2])
    ax_s = fig.add_subplot(gs[0, 3])
    ax_u = fig.add_subplot(gs[0, 4])
    panel_label(ax_he, "B", x=-0.22, y=1.08)
    show_he(ax_he, spatial, "Kidney tissue")
    show_gene(ax_slc, spatial, "Slc34a1", "Slc34a1 context", "YlOrRd")
    show_gene(ax_umod, spatial, "Umod", "Umod context", "Blues")
    for ax, gene in [(ax_s, "Slc34a1"), (ax_u, "Umod")]:
        row = df[df["target_gene"].eq(gene)].iloc[0]
        y = np.arange(len(methods))[::-1]
        vals = [row[m] for m in methods]
        for yi, val, color in zip(y, vals, colors):
            ax.plot([min(vals) - 0.01, val], [yi, yi], color=color, linewidth=1.0, alpha=0.9)
            ax.scatter(val, yi, s=30, color=color, edgecolor="#333333", linewidth=0.35)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if gene == "Slc34a1" else [], fontsize=6.2)
        pad = max(0.02, (max(vals) - min(vals)) * 0.55)
        ax.set_xlim(min(vals) - pad, max(vals) + pad)
        ax.set_xlabel("Held-out Pearson", fontsize=6.6)
        ax.set_title(gene, fontsize=7.6)
        ax.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
        ax.tick_params(axis="x", labelsize=6.0)
    fig.suptitle("Kidney main-task boundary: spatial context plus compact method comparison", fontsize=8.8, fontweight="bold", y=1.02)
    save_panel(fig, "Fig5B_kidney_main_task_boundary")


def fig5c_kidney_marker_followup() -> None:
    df = pd.read_csv(TARGETED_GENE)
    out = df[df["dataset"].eq("mouse_kidney_10x")].copy()
    out = out.sort_values("spot_pearson_source_mean", ascending=True)
    y = np.arange(len(out))
    fig, ax = plt.subplots(figsize=(3.0, 2.05))
    panel_label(ax, "G", x=-0.22, y=1.10)
    for yi, row in zip(y, out.itertuples(index=False)):
        ax.plot(
            [0.65, row.spot_pearson_source_mean],
            [yi, yi],
            color=PAPER_COLORS["teal"],
            linewidth=1.85,
            alpha=0.75,
        )
    ax.scatter(
        out["spot_pearson_source_mean"],
        y,
        s=40,
        color=PAPER_COLORS["teal"],
        edgecolor=PAPER_COLORS["dark"],
        linewidth=0.35,
        zorder=3,
    )
    ax.axvline(0.65, color=PAPER_COLORS["dark"], linewidth=0.55, linestyle=(0, (2.5, 2.5)))
    ax.set_yticks(y)
    ax.set_yticklabels(out["target_gene"], fontsize=6.2)
    ax.set_xlim(0.64, 0.95)
    ax.set_xlabel("Fitted-source Pearson", fontsize=6.4)
    ax.set_title("Kidney marker extension", fontsize=7.7)
    ax.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
    ax.tick_params(axis="x", labelsize=5.8)
    save_panel(fig, "Fig5G_kidney_marker_extension")


def fig5d_liver_annotation_boundary() -> None:
    df = pd.read_csv(EVIDENCE)
    liver = df[df["dataset"].str.contains("liver", case=False, na=False)].copy()
    liver = liver.sort_values("candidate_value")
    colors = [PAPER_COLORS["terracotta"] if v < 0 else PAPER_COLORS["teal"] for v in liver["candidate_value"]]
    fig, ax = plt.subplots(figsize=(6.8, 2.75))
    x = np.arange(len(liver))
    ax.axhline(0, color="#333333", linewidth=0.55)
    ax.bar(x, liver["candidate_value"], color=colors, edgecolor=PAPER_COLORS["dark"], linewidth=0.42, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(liver["task_or_target"])
    ax.set_ylabel("Pearson delta")
    ax.set_title("Liver annotation-boundary transfer is small and gene-dependent", loc="left", fontsize=8.6)
    ax.grid(axis="y", color="#dddddd", linewidth=0.45, alpha=0.75)
    panel_label(ax, "D")
    save_panel(fig, "Fig5D_liver_annotation_boundary")


def fig5e_sagittal_negative_control() -> None:
    df = pd.read_csv(EVIDENCE)
    brain = df[df["dataset"].str.contains("sagittal", case=False, na=False)].copy()
    brain["short_task"] = brain["task_or_target"].str.replace("Section", "S", regex=False)
    fig, ax = plt.subplots(figsize=(7.3, 2.75))
    x = np.arange(len(brain))
    ax.axhline(0, color="#333333", linewidth=0.55)
    ax.bar(x, brain["candidate_value"], color=PAPER_COLORS["slate"], edgecolor=PAPER_COLORS["dark"], linewidth=0.42, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(brain["short_task"], rotation=20, ha="right")
    ax.set_ylabel("Pearson delta")
    ax.set_title("Healthy sagittal brain acts as portability boundary / negative control", loc="left", fontsize=8.6)
    ax.grid(axis="y", color="#dddddd", linewidth=0.45, alpha=0.75)
    panel_label(ax, "E")
    save_panel(fig, "Fig5E_sagittal_negative_control")


def fig5f_transfer_delta_summary() -> None:
    df = pd.read_csv(EVIDENCE)
    keep = df[
        df["dataset"].str.contains("liver|sagittal", case=False, na=False)
        | df["evidence_block"].eq("marker_screen_followup")
    ].copy()
    keep["label"] = keep["dataset"].map(clean_dataset).str.replace("\n", " ", regex=False) + " | " + keep["task_or_target"]
    keep = keep.sort_values(["dataset", "candidate_value"], ascending=[True, True])
    y = np.arange(len(keep))
    colors = []
    for _, row in keep.iterrows():
        role = str(row["summary_role"])
        val = float(row["candidate_value"])
        if "negative" in role:
            colors.append(PAPER_COLORS["slate"])
        elif val >= 0:
            colors.append(PAPER_COLORS["teal"])
        else:
            colors.append(PAPER_COLORS["terracotta"])

    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.axvline(0, color="#333333", linewidth=0.65)
    x_clip_left = -0.035
    xlim = (-0.040, 0.0125)
    ax.axvline(x_clip_left, color="#777777", linewidth=0.55, linestyle=(0, (2, 2)), alpha=0.65)
    for yi, value, color in zip(y, keep["candidate_value"], colors):
        display_value = max(float(value), x_clip_left)
        marker = "<" if value < x_clip_left else "o"
        size = 48 if value < x_clip_left else 38
        ax.plot([0, display_value], [yi, yi], color=color, linewidth=1.2, alpha=0.9)
        ax.scatter(display_value, yi, s=size, marker=marker, color=color, edgecolor="#333333", linewidth=0.35, zorder=3)
        if value < x_clip_left:
            ax.text(
                x_clip_left + 0.0012,
                yi + 0.26,
                f"actual {float(value):.3f}",
                fontsize=5.7,
                color=color,
                ha="left",
                va="bottom",
            )
    ax.set_yticks(y)
    ax.set_yticklabels(keep["label"], fontsize=6.3)
    ax.set_xlim(*xlim)
    ax.set_xticks([-0.03, -0.02, -0.01, 0.00, 0.01])
    ax.set_xlabel("Pearson delta or candidate effect")
    ax.set_title("Transfer effects are task-specific and claim-calibrated", loc="left", fontsize=8.6)
    ax.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
    panel_label(ax, "F", x=-0.08, y=1.07)
    save_panel(fig, "Fig5F_cross_tissue_transfer_delta_lollipop")


def fig5f_evidence_role_matrix() -> None:
    df = pd.read_csv(EVIDENCE)

    block_order = [
        "primary_kidney_task",
        "prior_geometry_check",
        "marker_screen_followup",
        "annotation_boundary",
        "healthy_brain_portability",
    ]
    block_labels = {
        "primary_kidney_task": "Kidney\nmain task",
        "prior_geometry_check": "Kidney\nprior geometry",
        "marker_screen_followup": "Kidney\nmarker screen",
        "annotation_boundary": "Liver/APAP\nannotation",
        "healthy_brain_portability": "Sagittal brain\nnegative control",
    }
    role_order = [
        "method-development signal",
        "diagnostic / mixed",
        "small supplementary",
        "claim boundary",
        "negative control",
    ]
    role_colors = {
        "method-development signal": PAPER_COLORS["teal"],
        "diagnostic / mixed": PAPER_COLORS["slate"],
        "small supplementary": PAPER_COLORS["sage"],
        "claim boundary": PAPER_COLORS["terracotta"],
        "negative control": PAPER_COLORS["slate"],
    }

    def role_group(role: str) -> str:
        if "negative control" in role:
            return "negative control"
        if role == "method-development":
            return "method-development signal"
        if "diagnostic" in role or "mixed" in role:
            return "diagnostic / mixed"
        if "supplementary" in role:
            return "small supplementary"
        return "claim boundary"

    def effect_symbol(values: pd.Series) -> str:
        vals = values.astype(float)
        pos = int((vals > 0).sum())
        neg = int((vals < 0).sum())
        if pos and neg:
            return "+/-"
        if pos:
            return "+"
        if neg:
            return "-"
        return "0"

    grouped = (
        df.assign(role_group=df["summary_role"].map(role_group))
        .groupby(["evidence_block", "role_group"], sort=False)
        .agg(n=("task_or_target", "size"), value=("candidate_value", effect_symbol))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.7, 3.35))
    for x, block in enumerate(block_order):
        ax.axvline(x, color="#eeeeee", linewidth=0.65, zorder=0)
    for y, role in enumerate(role_order):
        ax.axhline(y, color="#eeeeee", linewidth=0.65, zorder=0)

    for _, row in grouped.iterrows():
        if row["evidence_block"] not in block_order or row["role_group"] not in role_order:
            continue
        x = block_order.index(row["evidence_block"])
        y = role_order.index(row["role_group"])
        n = int(row["n"])
        color = role_colors[row["role_group"]]
        ax.scatter(
            x,
            y,
            s=230 + 120 * n,
            color=color,
            edgecolor=PAPER_COLORS["dark"],
            linewidth=0.55,
            alpha=0.92,
            zorder=3,
        )
        ax.text(
            x,
            y,
            f"{row['value']}\n{n}",
            ha="center",
            va="center",
            fontsize=6.1,
            color="white" if row["role_group"] != "small supplementary" else PAPER_COLORS["dark"],
            fontweight="bold",
            zorder=4,
        )

    ax.set_xlim(-0.55, len(block_order) - 0.45)
    ax.set_ylim(len(role_order) - 0.55, -0.55)
    ax.set_xticks(range(len(block_order)))
    ax.set_xticklabels([block_labels[x] for x in block_order], fontsize=6.7)
    ax.set_yticks(range(len(role_order)))
    ax.set_yticklabels(role_order, fontsize=6.8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Cross-tissue evidence-role matrix", loc="left", fontsize=8.8, fontweight="bold", pad=9)
    ax.text(
        0.0,
        -0.22,
        "Cell text: effect direction (+/-/0) and number of evidence rows. Detailed values are retained in Supplementary Tables S6-S7.",
        transform=ax.transAxes,
        fontsize=6.3,
        color="#5b6470",
        ha="left",
        va="top",
    )
    save_panel(fig, "Fig5H_cross_tissue_evidence_role_matrix")


def fig5g_marker_rank_context() -> None:
    rank = pd.read_csv(MARKER_RANK).head(12).copy()
    rank["task"] = rank["target_gene"] + "-" + rank["barrier_compartment"]
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    y = np.arange(len(rank))[::-1]
    colors = [PAPER_COLORS["teal"] if bool(x) else PAPER_COLORS["slate"] for x in rank["recommended"]]
    ax.barh(y, rank["task_score"], height=0.48, color=colors, edgecolor=PAPER_COLORS["dark"], linewidth=0.42)
    ax.set_yticks(y)
    ax.set_yticklabels(rank["task"], fontsize=6.8)
    ax.set_xlabel("Task score")
    ax.set_title("Kidney systematic marker screen context", loc="left", fontsize=8.6)
    ax.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
    panel_label(ax, "G")
    save_panel(fig, "Fig5G_kidney_marker_rank_context")


def rough_assembly() -> None:
    stems = [
        "Fig5A_kidney_spatial_gene_context",
        "Fig5B_kidney_main_task_boundary",
        "Fig5C_liver_spatial_annotation_context",
        "Fig5D_sagittal_spatial_negative_control_context",
        "Fig5E_cross_tissue_task_matrix",
        "Fig5F_cross_tissue_transfer_delta_lollipop",
        "Fig5G_kidney_marker_extension",
        "Fig5H_cross_tissue_evidence_role_matrix",
    ]
    imgs = [Image.open(OUT_DIR / f"{stem}.png").convert("RGB") for stem in stems]
    thumb_w = 1700
    thumbs = []
    for img in imgs:
        scale = thumb_w / img.width
        thumbs.append(img.resize((thumb_w, int(img.height * scale)), Image.Resampling.LANCZOS))
    gutter = 110
    cols = 2
    row_specs = [(0, 2), (2, 4), (4, 6), (6, 8)]
    row_heights = [max(thumbs[i].height for i in range(start, end)) for start, end in row_specs]
    canvas_w = cols * thumb_w + (cols + 1) * gutter
    canvas_h = sum(row_heights) + (len(row_specs) + 1) * gutter
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    y = gutter
    for row_idx, (start, end) in enumerate(row_specs):
        x = gutter
        if end - start == 1:
            x = (canvas_w - thumb_w) // 2
        for idx in range(start, end):
            canvas.paste(thumbs[idx], (x, y))
            x += thumb_w + gutter
        y += row_heights[row_idx] + gutter
    canvas.save(OUT_DIR / "Figure5_rough_assembly.png", dpi=(EXPORT_DPI, EXPORT_DPI))
    canvas.save(OUT_DIR / "Figure5_rough_assembly.pdf", resolution=EXPORT_DPI)


def main() -> None:
    setup()
    ensure_dir(OUT_DIR)
    fig5a_kidney_spatial_context()
    fig5b_kidney_main_boundary()
    fig5c_liver_spatial_context()
    fig5d_sagittal_spatial_context()
    fig5a_dataset_task_matrix()
    fig5f_transfer_delta_summary()
    fig5c_kidney_marker_followup()
    fig5f_evidence_role_matrix()
    rough_assembly()
    print(f"Wrote Figure 5 panel assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
