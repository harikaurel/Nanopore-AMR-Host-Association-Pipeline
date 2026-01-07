#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
from collections import Counter
import re
import pandas as pd


# -----------------------------
# Step 4: aggregate isolate info per contig
# -----------------------------
def aggregate_isolate_per_contig(df_read_ctg_iso: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ctg_id, sub in df_read_ctg_iso.groupby("ctg_id"):
        labels = sub["isolate_label"].dropna().astype(str).tolist()
        n = len(labels)

        if n == 0:
            out.append({
                "ctg_id": ctg_id,
                "gt_isolate_top": pd.NA,
                "gt_isolate_top_pct": pd.NA,
                "gt_isolate_dist": pd.NA,
                "gt_isolate_n": 0,
            })
            continue

        cnt = Counter(labels)
        top_label, top_count = cnt.most_common(1)[0]
        top_pct = top_count / n

        dist = ";".join(
            f"{lab}:{(c/n):.6f}"
            for lab, c in sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
        )
        species = [lab.split("_", 1)[0] for lab in labels]   # kp_13_14 -> kp
        cnt_sp = Counter(species)
        top_sp, top_sp_count = cnt_sp.most_common(1)[0]

        out.append({
            "ctg_id": ctg_id,
            "gt_isolate_top": top_sp,
            "gt_isolate_top_pct": top_sp_count,
            "gt_isolate_dist": dist,
            "gt_isolate_n": n,
        })

    return pd.DataFrame(out)

def load_read_ctg(path: Path) -> pd.DataFrame:
    """
    Load read->ctg TSV that may or may not have a header.
    If header missing: assume first two columns are read_id, ctg_id.
    """
    # Try with header
    df = pd.read_csv(path, sep="\t", dtype=str)

    if {"read_id", "ctg_id"}.issubset(df.columns):
        return df[["read_id", "ctg_id"]]

    # Otherwise assume no header (2 columns)
    df2 = pd.read_csv(path, sep="\t", dtype=str, header=None)
    if df2.shape[1] < 2:
        raise ValueError(f"{path} must have at least 2 columns (read_id, ctg_id)")

    df2 = df2.iloc[:, :2].copy()
    df2.columns = ["read_id", "ctg_id"]
    return df2


# -----------------------------
# Kraken2 raw output parser
# -----------------------------
_taxid_re = re.compile(r"\s*\(taxid\s+\d+\)\s*$")

def clean_kraken_label(raw: str) -> str:
    """'Bacteria (taxid 2)' -> 'Bacteria'"""
    if raw is None or pd.isna(raw):
        return pd.NA
    s = str(raw).strip()
    s = _taxid_re.sub("", s).strip()
    return s if s else pd.NA


def load_kraken_raw_outputs(kraken_root: Path) -> pd.DataFrame:
    """
    Parse raw Kraken2 output files with rows like:
      C <seq_id> <taxon name (taxid N)> <length> <kmer-map...>

    We extract:
      ctg_id = column 2
      kraken_assignment_raw = column 3
      kraken_assignment = cleaned version (no taxid suffix)
    """
    # Collect likely kraken output files. Adjust patterns if needed.
    patterns = ["*.kraken", "*.kraken2", "*.txt", "*kraken*.tsv", "*kraken*"]
    files = []
    for pat in patterns:
        files.extend(list(kraken_root.rglob(pat)))
    files = sorted(set([f for f in files if f.is_file()]))

    if not files:
        raise FileNotFoundError(f"No Kraken output files found under: {kraken_root}")

    rows = {}
    for fp in files:
        # Read line-by-line (Kraken outputs can be large)
        try:
            with fp.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        # sometimes whitespace-delimited; fallback
                        parts = line.rstrip("\n").split()
                    if len(parts) < 3:
                        continue

                    # Kraken2 standard: status, seq_id, taxon_label, ...
                    status, seq_id, taxon_raw = parts[0], parts[1], parts[2]

                    # keep both C and U if you want; usually C only is fine
                    # if status != "C": continue

                    ctg_id = seq_id
                    if ctg_id not in rows:
                        rows[ctg_id] = {
                            "ctg_id": ctg_id,
                            "kraken_assignment_raw": taxon_raw,
                            "kraken_assignment": clean_kraken_label(taxon_raw),
                        }
        except Exception:
            continue

    if not rows:
        raise ValueError(
            "Found Kraken files but could not parse any rows. "
            "Expected Kraken2 raw format: C/U <id> <label (taxid)> ..."
        )

    return pd.DataFrame(list(rows.values()))


# -----------------------------
# MobSuite contig_report parser (your format)
# -----------------------------
def load_mobsuite_contig_report(mobsuite_root: Path) -> pd.DataFrame:
    """
    Your contig_report looks like a wide TSV with rows like:
      metaflye_polished  plasmid  ...  contig_102  49978  ...  Escherichia coli ...

    We only need:
      ctg_id (the token that matches 'contig_*')
      molecule_type_mock (2nd column in your example: plasmid/chromosome)

    This parser:
      - finds contig_report.txt recursively
      - for each line, takes molecule_type as column[1]
      - finds the first token matching /^contig_/ in the line as ctg_id
    """
    reports = sorted(mobsuite_root.rglob("contig_report.txt"))
    if not reports:
        raise FileNotFoundError(f"No contig_report.txt found under: {mobsuite_root}")

    ctg2type = {}

    for rpt in reports:
        with rpt.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    parts = line.rstrip("\n").split()
                if len(parts) < 2:
                    continue

                molecule_type = parts[1].strip()
                # find contig token
                ctg_id = None
                for tok in parts:
                    if tok.startswith("contig_"):
                        ctg_id = tok
                        break
                if ctg_id is None:
                    continue

                # first wins
                if ctg_id not in ctg2type:
                    ctg2type[ctg_id] = molecule_type

    if not ctg2type:
        raise ValueError("Parsed contig_report.txt files but found no contig_ IDs.")

    return pd.DataFrame({"ctg_id": list(ctg2type.keys()),
                         "molecule_type_mock": list(ctg2type.values())})


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Build contig-level evaluation table (GT + MobSuite + Kraken raw).")
    ap.add_argument("--read-isolate", required=True, type=Path,
                    help="TSV: read_id, isolate_label (Step 1)")
    ap.add_argument("--read-ctg", required=True, type=Path,
                    help="TSV: read_id, ctg_id (Step 2, primary-only mapping)")
    ap.add_argument("--mobsuite-root", required=True, type=Path,
                    help="Root dir containing nested MobSuite outputs with contig_report.txt")
    ap.add_argument("--kraken-root", required=True, type=Path,
                    help="Root dir containing nested Kraken2 raw output files")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output TSV")
    args = ap.parse_args()

    df_iso = pd.read_csv(args.read_isolate, sep="\t", dtype=str)
    df_ctg = load_read_ctg(args.read_ctg)

    if not {"read_id", "isolate_label"}.issubset(df_iso.columns):
        raise ValueError("read-isolate TSV must have columns: read_id, isolate_label")
    if not {"read_id", "ctg_id"}.issubset(df_ctg.columns):
        raise ValueError("read-ctg TSV must have columns: read_id, ctg_id")

    # join read->ctg with read->isolate
    df_join = df_ctg.merge(df_iso, on="read_id", how="left").dropna(subset=["isolate_label"])

    # GT per contig
    df_gt = aggregate_isolate_per_contig(df_join[["read_id", "ctg_id", "isolate_label"]])

    # MobSuite & Kraken
    df_mob = load_mobsuite_contig_report(args.mobsuite_root)
    df_kra = load_kraken_raw_outputs(args.kraken_root)

    # merge
    df = df_gt.merge(df_mob, on="ctg_id", how="left").merge(df_kra, on="ctg_id", how="left")

    # final column order (exactly what you listed)
    cols = [
        "ctg_id",
        "gt_isolate_top",
        "gt_isolate_dist",
        "molecule_type_mock",
        "kraken_assignment",
        "kraken_assignment_raw"
    ]
    df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]

    df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(df):,} contigs -> {args.out}")


if __name__ == "__main__":
    main()
