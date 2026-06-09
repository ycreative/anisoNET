"""Standardize GSE193107 mouse brain Visium archives for anisoNET.

Raw archives in a user-provided GSE193107 download directory contain Space Ranger-like
files nested under sample-specific prefixes. This script extracts one or more
samples into a consistent Visium directory layout under ``codexAnalysis``:

sample/
  filtered_feature_bc_matrix.h5
  raw_feature_bc_matrix.h5
  spatial/
    tissue_positions_list.csv
    scalefactors_json.json
    tissue_hires_image.png
    tissue_fullres_image.tif  (optional, decompressed from the matched tif.gz)
"""

from __future__ import annotations

import os

import argparse
import gzip
import json
import shutil
import tarfile
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RAW_DIR = Path(os.environ.get("ANISONET_GSE193107_RAW_DIR", PROJECT_ROOT / "dataset" / "GSE193107_RAW"))
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "processed_visium" / "brain_aging_gse193107"

SPACE_FILES = {
    "tissue_positions_list.csv",
    "scalefactors_json.json",
    "tissue_hires_image.png",
}

ROOT_FILES = {
    "filtered_feature_bc_matrix.h5",
    "raw_feature_bc_matrix.h5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize GSE193107 Visium archives.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory containing GSE193107 raw archives.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="Output root for standardized samples.")
    parser.add_argument("--sample", action="append", default=None, help="Sample archive stem to process. Repeatable.")
    parser.add_argument("--all", action="store_true", help="Process all GSM*.tar.gz archives in raw-dir.")
    parser.add_argument("--include-fullres-tif", action="store_true", help="Also decompress matched tif.gz images.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing standardized files.")
    return parser.parse_args()


def discover_samples(raw_dir: Path) -> list[str]:
    return sorted(path.name.removesuffix(".tar.gz") for path in raw_dir.glob("GSM*.tar.gz"))


def selected_samples(raw_dir: Path, requested: Iterable[str] | None, process_all: bool) -> list[str]:
    if process_all:
        return discover_samples(raw_dir)
    if requested:
        return list(requested)
    raise ValueError("Provide --sample at least once, or use --all")


def safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = tar.extractfile(member)
    if source is None:
        return
    with source, destination.open("wb") as handle:
        shutil.copyfileobj(source, handle)


def standardize_archive(
    archive_path: Path,
    output_dir: Path,
    *,
    include_fullres_tif: bool,
    overwrite: bool,
) -> dict:
    if output_dir.exists() and not overwrite:
        existing = output_dir / "filtered_feature_bc_matrix.h5"
        spatial_existing = output_dir / "spatial" / "tissue_hires_image.png"
        if existing.exists() and spatial_existing.exists():
            return {
                "sample": archive_path.name.removesuffix(".tar.gz"),
                "status": "skipped_existing",
                "output_dir": str(output_dir),
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_dir = output_dir / "spatial"
    spatial_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            basename = Path(member.name).name
            if basename in ROOT_FILES:
                safe_extract_member(tar, member, output_dir / basename)
                extracted.append(basename)
            elif basename in SPACE_FILES:
                safe_extract_member(tar, member, spatial_dir / basename)
                extracted.append(f"spatial/{basename}")

    tif_gz = archive_path.with_suffix("").with_suffix(".tif.gz")
    if include_fullres_tif and tif_gz.exists():
        tif_out = spatial_dir / "tissue_fullres_image.tif"
        if overwrite or not tif_out.exists():
            with gzip.open(tif_gz, "rb") as source, tif_out.open("wb") as handle:
                shutil.copyfileobj(source, handle)
        extracted.append("spatial/tissue_fullres_image.tif")

    required = [
        output_dir / "filtered_feature_bc_matrix.h5",
        spatial_dir / "tissue_positions_list.csv",
        spatial_dir / "scalefactors_json.json",
        spatial_dir / "tissue_hires_image.png",
    ]
    hires_png = spatial_dir / "tissue_hires_image.png"
    lowres_png = spatial_dir / "tissue_lowres_image.png"
    if hires_png.exists() and (overwrite or not lowres_png.exists()):
        shutil.copyfile(hires_png, lowres_png)
        extracted.append("spatial/tissue_lowres_image.png")

    missing = [str(path) for path in required if not path.exists()]
    status = "ok" if not missing else "missing_required_files"

    return {
        "sample": archive_path.name.removesuffix(".tar.gz"),
        "status": status,
        "output_dir": str(output_dir),
        "extracted": extracted,
        "missing": missing,
    }


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for sample in selected_samples(raw_dir, args.sample, args.all):
        archive_path = raw_dir / f"{sample}.tar.gz"
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        result = standardize_archive(
            archive_path,
            output_root / sample,
            include_fullres_tif=args.include_fullres_tif,
            overwrite=args.overwrite,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    manifest_path = output_root / "standardization_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()

