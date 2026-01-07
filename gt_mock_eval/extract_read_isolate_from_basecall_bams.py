#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1: Extract read IDs and isolate labels from UNALIGNED Dorado basecalling BAMs.

Assumption:
- Each BAM file corresponds to one isolate.
- The BAM filename (without .bam) is used as the isolate label.

Example:
  ecoli_37_38.bam -> isolate_label = "ecoli_37_38"
"""

from pathlib import Path
import argparse
import pysam
import pandas as pd


def iter_read_ids(bam_path: Path):
    """Yield read IDs from an unaligned BAM."""
    with pysam.AlignmentFile(bam_path, check_sq=False) as bam:
        for rec in bam.fetch(until_eof=True):
            yield rec.query_name


def main():
    parser = argparse.ArgumentParser(
        description="Extract read_id → isolate_label from unaligned Dorado basecalling BAM files."
    )
    parser.add_argument(
        "--bam-dir",
        required=True,
        type=Path,
        help="Directory containing unaligned Dorado basecalling *.bam files.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output TSV file (columns: read_id, isolate_label).",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Drop duplicate read_id entries (keep first occurrence).",
    )

    args = parser.parse_args()

    bam_files = sorted(args.bam_dir.glob("*.bam"))
    if not bam_files:
        raise FileNotFoundError(f"No *.bam files found in: {args.bam_dir}")

    rows = []
    for bam_path in bam_files:
        isolate_label = bam_path.stem
        print(f"Processing {bam_path.name} → isolate_label={isolate_label}")

        for read_id in iter_read_ids(bam_path):
            rows.append({
                "read_id": read_id,
                "isolate_label": isolate_label
            })

    df = pd.DataFrame(rows)

    if args.deduplicate and not df.empty:
        before = len(df)
        df = df.drop_duplicates(subset=["read_id"], keep="first")
        print(f"Deduplicated: {before:,} → {len(df):,}")

    df.to_csv(args.out, sep="\t", index=False)
    print(f"Done. Rows: {len(df):,}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
