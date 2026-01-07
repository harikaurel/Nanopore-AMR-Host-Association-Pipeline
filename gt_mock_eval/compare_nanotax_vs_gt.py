#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd

# If your gt_isolate_top is a code like "smarcescens_35_36",
# this maps the prefix to a full species name.
SPECIES_MAP = {
    "cfreundii": "Citrobacter freundii",
    "rplanticola": "Raoultella planticola",
    "ecoli": "Escherichia coli",
    "pmirabilis": "Proteus mirabilis",
    "smarcescens": "Serratia marcescens",
    "klox": "Klebsiella oxytoca",
    "kp": "Klebsiella pneumoniae",
    "kpneumoniae": "Klebsiella pneumoniae",
}

PROFILE_COL_CANDIDATES = [
    "nanomotif_taxonomic_association_profile",
    "nanomotif_taxonomic_association_profile",  # common alt spelling
    "nanomotif_taxonomic",  # just in case
    "nanomotif_profile",
]


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find any of columns {candidates} in file. Columns={list(df.columns)}"
    )


def normalize_name(s: str) -> str:
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s).strip())


def gt_to_species(gt_isolate_top: str) -> str:
    """
    If gt_isolate_top is already a full species name, keep it.
    If it's a code like 'smarcescens_35_36', map prefix -> species name.
    """
    if pd.isna(gt_isolate_top):
        return ""
    s = str(gt_isolate_top).strip()
    if not s:
        return ""

    # If it already looks like a binomial, keep as-is
    parts = s.split()
    if len(parts) >= 2:
        return " ".join(parts[:2])

    # Otherwise treat it as code (possibly with _barcode)
    code = s.split("_")[0].lower()
    return SPECIES_MAP.get(code, s)  # fallback: keep original


_profile_item_re = re.compile(r"^\s*(.+?)\s*\(\s*(\d+)\s*\)\s*$")


def top_taxon_from_profile(profile: str) -> str:
    """
    Parse a string like:
      "Citrobacter(18); Citrobacter freundii(14); Bacteria(1)"
    Return the taxon with the highest count (ties: first encountered).
    If parsing fails, return the first token before ';' (stripped).
    """
    if pd.isna(profile):
        return ""
    s = str(profile).strip()
    if not s:
        return ""

    best_tax, best_n = "", -1
    for token in s.split(";"):
        token = token.strip()
        if not token:
            continue
        m = _profile_item_re.match(token)
        if m:
            tax = m.group(1).strip()
            n = int(m.group(2))
            if n > best_n:
                best_tax, best_n = tax, n
        else:
            # If no "(n)" format, treat as a candidate with count 0
            if best_n < 0:
                best_tax, best_n = token, 0

    return best_tax


def binomial(name: str) -> str:
    """
    Reduce a taxon label to its binomial (first two words), lowercased.
    Examples:
      "Escherichia coli O157" -> "escherichia coli"
      "Klebsiella pneumoniae subsp. pneumoniae 1084" -> "klebsiella pneumoniae"
      "Bacteria" -> "bacteria"
    """
    s = normalize_name(name)
    if not s:
        return ""
    parts = s.split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}".lower()
    return parts[0].lower()


def genus(name: str) -> str:
    s = normalize_name(name)
    return s.split()[0].lower() if s else ""


def main():
    ap = argparse.ArgumentParser(
        description="Compare Nanomotif taxonomic association profile (top taxon) to GT isolate top (binomial-aware)."
    )
    ap.add_argument(
        "--assoc-tsv",
        required=True,
        type=Path,
        help="Plasmid host association TSV with ctg_id and nanomotif_taxonomic_association_profile.",
    )
    ap.add_argument(
        "--gt-tsv",
        required=True,
        type=Path,
        help="Ground-truth contig table TSV with ctg_id and gt_isolate_top (or isolate_label, etc.).",
    )
    ap.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output TSV (4 columns: ctg_id, gt_species, nanomotif_taxon, correct).",
    )
    ap.add_argument(
        "--allow-genus-match",
        action="store_true",
        help="If set: count as correct when genus matches (useful if Nanomotif returns only genus).",
    )
    args = ap.parse_args()

    assoc = pd.read_csv(args.assoc_tsv, sep="\t", dtype=str)
    gt = pd.read_csv(args.gt_tsv, sep="\t", dtype=str)

    if "ctg_id" not in assoc.columns:
        raise ValueError(f"--assoc-tsv must have ctg_id. Columns={list(assoc.columns)}")
    if "ctg_id" not in gt.columns:
        raise ValueError(f"--gt-tsv must have ctg_id. Columns={list(gt.columns)}")

    prof_col = pick_col(assoc, PROFILE_COL_CANDIDATES)

    # find gt isolate column
    gt_col_candidates = ["gt_isolate_top", "gt_species", "isolate_species_barcode (pct)", "isolate_label"]
    gt_isolate_col = None
    for c in gt_col_candidates:
        if c in gt.columns:
            gt_isolate_col = c
            break
    if gt_isolate_col is None:
        raise ValueError(
            f"--gt-tsv must contain one of {gt_col_candidates}. Columns={list(gt.columns)}"
        )

    # Build comparison table
    assoc_small = assoc[["ctg_id", prof_col]].copy()
    assoc_small["nanomotif_taxon"] = assoc_small[prof_col].map(top_taxon_from_profile).map(normalize_name)

    gt_small = gt[["ctg_id", gt_isolate_col]].copy()
    gt_small["gt_species"] = gt_small[gt_isolate_col].map(gt_to_species).map(normalize_name)

    df = assoc_small.merge(gt_small[["ctg_id", "gt_species"]], on="ctg_id", how="left")

    # Binomial-aware correctness:
    # exact species match if first two tokens match (case-insensitive),
    # with optional genus fallback.
    df["gt_binomial"] = df["gt_species"].map(binomial)
    df["nm_binomial"] = df["nanomotif_taxon"].map(binomial)

    if args.allow_genus_match:
        df["correct"] = (
            (df["gt_binomial"] != "")
            & (
                (df["gt_binomial"] == df["nm_binomial"])
                | ((df["nm_binomial"] != "") & (df["nm_binomial"].map(lambda x: x.split()[0]) == df["gt_binomial"].map(lambda x: x.split()[0])))
            )
        )
    else:
        df["correct"] = (df["gt_binomial"] != "") & (df["gt_binomial"] == df["nm_binomial"])

    out_df = df[["ctg_id", "gt_species", "nanomotif_taxon", "correct"]].copy()
    out_df.to_csv(args.out, sep="\t", index=False)

    # Accuracy summary (only rows with GT present)
    comparable = out_df["gt_species"].astype(bool)
    n_total = len(out_df)
    n_comp = int(comparable.sum())
    n_correct = int(out_df.loc[comparable, "correct"].sum())
    acc = (n_correct / n_comp) if n_comp else 0.0

    print(f"[DONE] Wrote: {args.out}")
    print(f"Total rows: {n_total}")
    print(f"Rows with GT available: {n_comp}")
    print(f"Correct: {n_correct}")
    print(f"Accuracy (on rows with GT): {acc:.4f}")


if __name__ == "__main__":
    main()
