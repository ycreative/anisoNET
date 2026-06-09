"""Standardize GEO-prefixed 10x Visium files into the anisoNET layout.

Some GEO Visium deposits store files as, for example:

  GSMxxxx_filtered_feature_bc_matrix_SAMPLE.h5
  GSMxxxx_tissue_positions_list_SAMPLE.csv.gz
  GSMxxxx_scalefactors_json_SAMPLE.json.gz
  GSMxxxx_tissue_hires_image_SAMPLE.png.gz

This workflow normalizes those files to the lightweight anisoNET layout used by
`anisonet.visium_io.load_visium_lite`.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path


REQUIRED_KINDS = {
    "matrix": "filtered_feature_bc_matrix",
    "positions": "tissue_positions",
    "scalefactors": "scalefactors_json",
    "hires": "tissue_hires_image",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize GEO-prefixed 10x Visium files.")
    parser.add_argument("--input-dir", required=True, help="Directory containing prefixed GEO Visium files.")
    parser.add_argument("--sample-prefix", required=True, help="Prefix used to identify a sample, e.g. GSM8599602.")
    parser.add_argument("--output-dir", required=True, help="Standardized sample output directory.")
    parser.add_argument("--sample-name", default="", help="Optional sample label written to manifest.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_one(input_dir: Path, sample_prefix: str, token: str) -> Path:
    matches = sorted(input_dir.glob(f"{sample_prefix}*{token}*"))
    if not matches:
        raise FileNotFoundError(f"No file matching {sample_prefix}*{token}* in {input_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {sample_prefix}*{token}*: {matches}")
    return matches[0]


def copy_or_decompress(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".gz":
        with gzip.open(src, "rb") as source, dst.open("wb") as target:
            shutil.copyfileobj(source, target)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    spatial_dir = output_dir / "spatial"
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_dir.mkdir(parents=True, exist_ok=True)

    files = {kind: find_one(input_dir, args.sample_prefix, token) for kind, token in REQUIRED_KINDS.items()}

    copy_or_decompress(files["matrix"], output_dir / "filtered_feature_bc_matrix.h5", overwrite=args.overwrite)
    copy_or_decompress(files["positions"], spatial_dir / "tissue_positions_list.csv", overwrite=args.overwrite)
    copy_or_decompress(files["scalefactors"], spatial_dir / "scalefactors_json.json", overwrite=args.overwrite)
    copy_or_decompress(files["hires"], spatial_dir / "tissue_hires_image.png", overwrite=args.overwrite)

    # The lightweight reader only needs the hires image, but keeping a lowres alias
    # makes the directory compatible with tools that expect Space Ranger output.
    copy_or_decompress(files["hires"], spatial_dir / "tissue_lowres_image.png", overwrite=args.overwrite)

    manifest = output_dir / "standardization_manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                f"sample_name={args.sample_name}",
                f"sample_prefix={args.sample_prefix}",
                f"input_dir={input_dir.resolve()}",
                *[f"{kind}={path.resolve()}" for kind, path in files.items()],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Standardized GEO-prefixed Visium sample to {output_dir}")


if __name__ == "__main__":
    main()
