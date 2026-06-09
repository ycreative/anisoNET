"""Draw a controlled vector-style Figure 1 method overview draft."""

from __future__ import annotations

import os

from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUTPUT_DIR = PROJECT_ROOT / "codexAnalysis" / "manuscript_figures" / "drafts"


def main() -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "input": "#e9f2f7",
        "prior": "#f2eadf",
        "model": "#eef0e4",
        "output": "#f5e8ec",
        "stroke": "#263238",
        "accent": "#2a9d8f",
    }

    # Panel bands
    panel_specs = [
        ("A", "Spatial inputs", 0.04, 0.57, 0.20, 0.34, colors["input"]),
        ("B", "Priors on a common grid", 0.29, 0.57, 0.30, 0.34, colors["prior"]),
        ("C", "Scalar PINN field model", 0.64, 0.57, 0.32, 0.34, colors["model"]),
        ("D", "Tissue-constrained outputs", 0.12, 0.10, 0.76, 0.31, colors["output"]),
    ]
    for label, title, x, y, w, h, color in panel_specs:
        draw_box(ax, x, y, w, h, color=color, edge=colors["stroke"], linewidth=0.8)
        ax.text(x + 0.014, y + h - 0.035, label, weight="bold", fontsize=10, color=colors["stroke"])
        ax.text(x + 0.048, y + h - 0.031, title, weight="bold", fontsize=8.5, color=colors["stroke"])

    # A: inputs
    draw_mini_tissue(ax, 0.075, 0.70, 0.075, 0.095, colors["accent"])
    ax.text(0.113, 0.675, "H&E image", ha="center", va="top")
    draw_spot_grid(ax, 0.165, 0.70, 0.075, 0.095)
    ax.text(0.202, 0.675, "Visium spots\nand counts", ha="center", va="top")

    # B: priors
    prior_items = [
        ("Source\nS(x,y)", 0.325, 0.745, "#b83b5e"),
        ("Barrier\nB(x,y)", 0.425, 0.745, "#d95f02"),
        ("Histology\nH(x,y)", 0.525, 0.745, "#6c757d"),
        ("Diffusion\nD(x,y)", 0.425, 0.625, "#457b9d"),
    ]
    for text, cx, cy, color in prior_items:
        draw_grid_icon(ax, cx, cy, color)
        ax.text(cx, cy - 0.065, text, ha="center", va="top")
    ax.text(
        0.425,
        0.585,
        "D = D_H exp(-alpha B)\nscalar coefficient",
        ha="center",
        va="top",
        fontsize=7,
        color=colors["stroke"],
    )

    # C: model
    ax.text(
        0.80,
        0.805,
        "C_theta(x,y)",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        color=colors["stroke"],
    )
    ax.text(
        0.80,
        0.745,
        "D(x,y)(C_xx + C_yy) - kC + S(x,y) = 0",
        ha="center",
        va="center",
        fontsize=7.5,
        color=colors["stroke"],
    )
    loss_text = (
        "Loss = data + PDE residual\n"
        "+ boundary + background + smoothness"
    )
    ax.text(0.80, 0.675, loss_text, ha="center", va="center", fontsize=7)
    ax.text(
        0.80,
        0.615,
        "Current implementation:\nscalar diffusion, not tensor",
        ha="center",
        va="center",
        fontsize=7,
        color="#8a1c1c",
    )

    # D: outputs
    output_items = [
        ("Raw field", 0.22, 0.25, "#b83b5e"),
        ("Tissue mask", 0.39, 0.25, "#6c757d"),
        ("Masked field", 0.56, 0.25, "#2a9d8f"),
        ("Barrier-aware\navailability", 0.73, 0.25, "#457b9d"),
    ]
    for text, cx, cy, color in output_items:
        draw_grid_icon(ax, cx, cy, color)
        ax.text(cx, cy - 0.065, text, ha="center", va="top")
    ax.text(
        0.50,
        0.135,
        "Quantify fitted source agreement, roughness, leakage, and sensitivity; avoid generic interpolation claims.",
        ha="center",
        va="center",
        fontsize=7,
        color=colors["stroke"],
    )

    # Arrows
    arrow(ax, (0.245, 0.745), (0.295, 0.745), colors["stroke"])
    arrow(ax, (0.590, 0.745), (0.640, 0.745), colors["stroke"])
    arrow(ax, (0.795, 0.570), (0.700, 0.410), colors["stroke"])
    arrow(ax, (0.425, 0.610), (0.650, 0.700), colors["stroke"], connectionstyle="arc3,rad=-0.15")

    ax.text(
        0.50,
        0.965,
        "anisoNET: scalar physics-informed field inference for anatomy-aware spatial transcriptomics",
        ha="center",
        va="top",
        fontsize=11,
        weight="bold",
        color=colors["stroke"],
    )

    fig.savefig(OUTPUT_DIR / "Figure1_method_overview_draft.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "Figure1_method_overview_draft.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, x, y, w, h, *, color, edge, linewidth=0.6):
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=edge, linewidth=linewidth))


def arrow(ax, start, end, color, *, connectionstyle="arc3,rad=0"):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def draw_grid_icon(ax, cx, cy, color):
    import numpy as np
    from matplotlib.patches import Rectangle

    size = 0.064
    ax.add_patch(Rectangle((cx - size / 2, cy - size / 2), size, size, facecolor="white", edgecolor="#263238", linewidth=0.5))
    xs = np.linspace(cx - size / 2 + 0.004, cx + size / 2 - 0.004, 6)
    ys = np.linspace(cy - size / 2 + 0.004, cy + size / 2 - 0.004, 6)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            alpha = 0.25 + 0.65 * ((i + j) / 10)
            ax.scatter(x, y, s=5, color=color, alpha=alpha, linewidths=0)


def draw_mini_tissue(ax, x, y, w, h, color):
    import numpy as np

    t = np.linspace(0, 2 * np.pi, 80)
    rx = w * (0.46 + 0.08 * np.sin(3 * t))
    ry = h * (0.43 + 0.07 * np.cos(4 * t))
    ax.fill(x + w / 2 + rx * np.cos(t), y + h / 2 + ry * np.sin(t), color=color, alpha=0.35, edgecolor="#263238", linewidth=0.5)
    ax.scatter([x + 0.03, x + 0.05, x + 0.045], [y + 0.06, y + 0.04, y + 0.075], s=6, color="#b83b5e", alpha=0.8)


def draw_spot_grid(ax, x, y, w, h):
    import numpy as np

    xs = np.linspace(x + 0.012, x + w - 0.012, 4)
    ys = np.linspace(y + 0.016, y + h - 0.016, 4)
    for xx in xs:
        for yy in ys:
            ax.scatter(xx, yy, s=10, facecolor="#f4a261", edgecolor="#263238", linewidth=0.3)


if __name__ == "__main__":
    main()

