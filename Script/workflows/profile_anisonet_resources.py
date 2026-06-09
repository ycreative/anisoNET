"""Profile anisoNET PINN runtime and GPU memory on representative inputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"K:\YC\experiment\STagent")
SCRIPT_ROOT = PROJECT_ROOT / "Script"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anisonet.pinn import get_profile, robust_normalize, solve_scalar_reaction_diffusion
from anisonet.postprocessing import mask_field, normalize_masked_field, smooth_inside_mask
from anisonet.preprocessing import clip_and_normalize
from anisonet.visium_io import load_visium_lite, normalized_gene_vector


PROCESSED_ROOT = PROJECT_ROOT / "codexAnalysis" / "processed_visium" / "brain_aging_gse193107"
PREFLIGHT_ROOT = PROJECT_ROOT / "codexAnalysis" / "histology_prior" / "brain_aging_gse193107"
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "resource_profile" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile anisoNET runtime and GPU memory.")
    parser.add_argument("--sample", default="GSM5773457_Old_mouse_brain_A1-2")
    parser.add_argument("--target-gene", action="append", default=["Apoe", "Gfap"])
    parser.add_argument("--barrier-name", default="CNS_Myelin")
    parser.add_argument("--histology-prior", default="brightness")
    parser.add_argument("--profile", default="fourier_refined_16g")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--postprocess-sigma", type=float, default=0.7)
    parser.add_argument("--source-percentile", type=float, default=99.0)
    parser.add_argument("--output-percentile", type=float, default=99.5)
    parser.add_argument("--noise-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix for summary files, e.g. low_pde_candidate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = OUTPUT_ROOT / args.sample
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    hardware = get_hardware(args.device)
    for target_gene in args.target_gene:
        rows.append(profile_one(args, target_gene, output_dir, hardware))
    frame = pd.DataFrame(rows)
    suffix = sanitize_suffix(args.output_suffix)
    profile_csv = output_dir / f"anisonet_resource_profile{suffix}.csv"
    frame.to_csv(profile_csv, index=False)
    write_interpretation(frame, hardware, output_dir, suffix=suffix)
    print(frame.to_string(index=False))


def profile_one(args: argparse.Namespace, target_gene: str, output_dir: Path, hardware: dict[str, object]) -> dict[str, object]:
    import torch

    sample_dir = PROCESSED_ROOT / args.sample
    preflight_dir = PREFLIGHT_ROOT / args.sample / f"{target_gene}_{args.barrier_name}" / args.histology_prior
    run_dir = output_dir / f"{target_gene}_{args.barrier_name}_{args.histology_prior}_{args.profile}"
    run_dir.mkdir(parents=True, exist_ok=True)

    coords_norm = np.load(preflight_dir / "coords_norm.npy")
    source_grid = np.load(preflight_dir / "source_grid.npy")
    diffusion_grid = np.load(preflight_dir / "diffusion_grid.npy")
    tissue_mask = np.load(preflight_dir / "tissue_mask.npy")

    sample = load_visium_lite(sample_dir)
    source_values = clip_and_normalize(
        normalized_gene_vector(sample, target_gene),
        percentile=args.source_percentile,
    )
    profile = get_profile(args.profile)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = solve_scalar_reaction_diffusion(
        coords_norm,
        source_values,
        diffusion_grid,
        source_grid,
        profile=profile,
        tissue_mask=tissue_mask,
        device=args.device,
        seed=args.seed,
        prediction_grid_size=diffusion_grid.shape[0],
    )
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
    else:
        peak_allocated_gb = float("nan")
        peak_reserved_gb = float("nan")
    elapsed_seconds = time.perf_counter() - start

    spot_norm = robust_normalize(result.spot_prediction, percentile=args.output_percentile)
    grid_norm = robust_normalize(result.grid_prediction, percentile=args.output_percentile)
    grid_clean = np.where(grid_norm < args.noise_threshold, 0.0, grid_norm)
    grid_masked = mask_field(grid_clean, tissue_mask)
    grid_postprocessed = None
    if args.postprocess_sigma > 0:
        grid_smoothed = smooth_inside_mask(grid_masked, tissue_mask, sigma=args.postprocess_sigma)
        grid_postprocessed = normalize_masked_field(grid_smoothed, tissue_mask, percentile=args.output_percentile)

    np.save(run_dir / "pinn_spot_prediction_norm.npy", spot_norm.astype(np.float32))
    np.save(run_dir / "pinn_grid_prediction_clean_tissue_masked.npy", grid_masked.astype(np.float32))
    if grid_postprocessed is not None:
        np.save(run_dir / "pinn_grid_prediction_postprocessed.npy", grid_postprocessed.astype(np.float32))
    with (run_dir / "pinn_history.json").open("w", encoding="utf-8") as handle:
        json.dump(result.history, handle, indent=2)

    row = {
        "sample": args.sample,
        "target_gene": target_gene,
        "barrier_name": args.barrier_name,
        "histology_prior": args.histology_prior,
        "profile": args.profile,
        "device_requested": args.device,
        "cuda_available": bool(hardware["cuda_available"]),
        "gpu_name": hardware["gpu_name"],
        "gpu_total_memory_gb": hardware["gpu_total_memory_gb"],
        "torch_version": hardware["torch_version"],
        "n_spots": int(coords_norm.shape[0]),
        "grid_size": int(diffusion_grid.shape[0]),
        "hidden_width": int(profile.hidden_width),
        "hidden_depth": int(profile.hidden_depth),
        "network": profile.network,
        "fourier_features": int(profile.fourier_features),
        "num_domain": int(profile.num_domain),
        "num_boundary": int(profile.num_boundary),
        "iterations": int(profile.iterations),
        "elapsed_seconds": float(elapsed_seconds),
        "elapsed_minutes": float(elapsed_seconds / 60.0),
        "peak_cuda_allocated_gb": float(peak_allocated_gb),
        "peak_cuda_reserved_gb": float(peak_reserved_gb),
        "reserved_fraction_of_gpu": float(peak_reserved_gb / hardware["gpu_total_memory_gb"]) if hardware["gpu_total_memory_gb"] else float("nan"),
        "output_dir": str(run_dir),
    }
    with (run_dir / "resource_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2)
    return row


def get_hardware(device: str) -> dict[str, object]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = None
    gpu_total_memory_gb = None
    if device.startswith("cuda") and cuda_available:
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        gpu_name = props.name
        gpu_total_memory_gb = props.total_memory / 1024**3
    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_total_memory_gb": gpu_total_memory_gb,
    }


def sanitize_suffix(value: str) -> str:
    if not value:
        return ""
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return f"_{safe}" if safe else ""


def write_interpretation(frame: pd.DataFrame, hardware: dict[str, object], output_dir: Path, *, suffix: str = "") -> None:
    mean_minutes = frame["elapsed_minutes"].mean()
    max_reserved = frame["peak_cuda_reserved_gb"].max()
    total = hardware["gpu_total_memory_gb"]
    fraction = max_reserved / total if total else np.nan
    lines = [
        "# anisoNET Resource Profile",
        "",
        f"Hardware: `{hardware['gpu_name']}` with `{total:.2f}` GB GPU memory; PyTorch `{hardware['torch_version']}`.",
        "",
        "## Representative Run",
        "",
        f"- Profiled `{len(frame)}` target genes on one representative old brain section.",
        f"- Mean PINN runtime was `{mean_minutes:.2f}` minutes per target gene.",
        f"- Maximum CUDA reserved memory was `{max_reserved:.2f}` GB, or `{fraction:.1%}` of available GPU memory.",
        "",
        "## Interpretation",
        "",
        f"The profiled `{frame['profile'].iloc[0]}` profile fits comfortably within a 16GB RTX 5060 Ti for single-gene field inference on this Visium-scale section. Full batches are feasible locally when run sequentially, while broad hyperparameter sweeps or many concurrent jobs remain better suited to a larger server GPU.",
        "",
    ]
    (output_dir / f"anisonet_resource_profile_interpretation{suffix}.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
