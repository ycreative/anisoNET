"""Standardize public 10x Visium files into the lightweight anisoNET layout.

Expected output layout:

sample_dir/
  filtered_feature_bc_matrix.h5
  spatial/
    tissue_positions_list.csv
    scalefactors_json.json
    tissue_hires_image.png
    tissue_lowres_image.png
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize public 10x Visium outputs for anisoNET.")
    parser.add_argument("--matrix-h5", required=True, help="Input filtered_feature_bc_matrix h5 file.")
    parser.add_argument("--output-dir", required=True, help="Standardized sample output directory.")
    parser.add_argument("--spatial-tar-gz", help="Input spatial.tar.gz file.")
    parser.add_argument("--spatial-dir", help="Input directory containing 10x spatial files.")
    parser.add_argument("--sample-name", default="", help="Optional sample name written to manifest only.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_or_decompress(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".gz":
        with gzip.open(src, "rb") as source, dst.open("wb") as target:
            shutil.copyfileobj(source, target)
    else:
        shutil.copy2(src, dst)


def standardize_spatial_from_dir(src_dir: Path, dst_dir: Path, *, overwrite: bool) -> None:
    aliases = {
        "tissue_positions_list.csv": ("tissue_positions_list.csv", "tissue_positions.csv"),
        "scalefactors_json.json": ("scalefactors_json.json",),
        "tissue_hires_image.png": ("tissue_hires_image.png",),
        "tissue_lowres_image.png": ("tissue_lowres_image.png",),
    }
    for dst_name, candidates in aliases.items():
        found = None
        for candidate in candidates:
            direct = src_dir / candidate
            gz = src_dir / f"{candidate}.gz"
            if direct.exists():
                found = direct
                break
            if gz.exists():
                found = gz
                break
        if found is None:
            raise FileNotFoundError(f"Missing spatial file for {dst_name} in {src_dir}")
        copy_or_decompress(found, dst_dir / dst_name, overwrite=overwrite)


def standardize_spatial_from_tar(tar_path: Path, dst_dir: Path, *, overwrite: bool) -> None:
    needed = {
        "tissue_positions_list.csv",
        "scalefactors_json.json",
        "tissue_hires_image.png",
        "tissue_lowres_image.png",
    }
    with tarfile.open(tar_path, "r:gz") as handle:
        members = {Path(member.name).name: member for member in handle.getmembers()}
        missing = sorted(name for name in needed if name not in members)
        if missing:
            raise FileNotFoundError(f"Missing spatial files in {tar_path}: {missing}")
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in sorted(needed):
            target = dst_dir / name
            if target.exists() and not overwrite:
                continue
            source = handle.extractfile(members[name])
            if source is None:
                raise FileNotFoundError(f"Could not extract {name} from {tar_path}")
            with source, target.open("wb") as out:
                shutil.copyfileobj(source, out)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    spatial_dir = output_dir / "spatial"
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_file(Path(args.matrix_h5), output_dir / "filtered_feature_bc_matrix.h5", overwrite=args.overwrite)
    if args.spatial_tar_gz:
        standardize_spatial_from_tar(Path(args.spatial_tar_gz), spatial_dir, overwrite=args.overwrite)
    elif args.spatial_dir:
        standardize_spatial_from_dir(Path(args.spatial_dir), spatial_dir, overwrite=args.overwrite)
    else:
        raise ValueError("Provide either --spatial-tar-gz or --spatial-dir")

    manifest = output_dir / "standardization_manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                f"sample_name={args.sample_name}",
                f"matrix_h5={Path(args.matrix_h5).resolve()}",
                f"spatial_tar_gz={Path(args.spatial_tar_gz).resolve() if args.spatial_tar_gz else ''}",
                f"spatial_dir={Path(args.spatial_dir).resolve() if args.spatial_dir else ''}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Standardized Visium sample to {output_dir}")


if __name__ == "__main__":
    main()

