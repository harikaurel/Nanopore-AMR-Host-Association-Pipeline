#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re
from collections import Counter
import csv
import sys

csv.field_size_limit(sys.maxsize)

MIN_OVERLAP_MOTIFS = 1
ATOL_TIE = 1e-12

TAXID_RE = re.compile(r"\(taxid\s+(\d+)\)")

# --------------------------------------------------
# helpers
# --------------------------------------------------
def parse_taxid_from_kraken_out_field(field: str) -> str:
    if not isinstance(field, str):
        return ""
    s = field.strip()
    # case A: "(taxid N)"
    m = TAXID_RE.search(s)
    if m:
        return m.group(1)
    # case B: field is just the numeric taxid
    if s.isdigit():
        return s
    return ""

def find_one(folder: Path, patterns: list[str], label: str) -> Path:
    hits = []
    for pat in patterns:
        hits.extend(folder.glob(pat))
    hits = [h for h in hits if h.is_file()]
    if not hits:
        raise SystemExit(f"[STOP] Could not find {label} in {folder}")
    if len(hits) > 1:
        print(f"[WARN] Multiple {label} files found, using {hits[0].name}")
    return hits[0]


def clean_taxon(label: str) -> str:
    if not isinstance(label, str):
        return ""
    s = re.sub(r"\([^)]*\)", "", label)
    return re.sub(r"\s+", " ", s).strip()


def rmsd_overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    m = (~np.isnan(a)) & (~np.isnan(b))
    n = int(m.sum())
    if n == 0:
        return float("inf"), 0
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d * d))), n


# --------------------------------------------------
# loaders
# --------------------------------------------------
def load_motif_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["median"] = df["median"].astype(float)
    df = df.groupby(["contig", "motif"], as_index=False)["median"].median()
    X = df.pivot(index="contig", columns="motif", values="median")
    X.index = X.index.astype(str)
    print(f"[INFO] Loaded Nanomotif vectors: {X.shape[0]} contigs")
    return X


def extract_contig_types(contig_report: Path) -> pd.DataFrame:
    df = pd.read_csv(contig_report, sep="\t", dtype=str)
    df["contig_id"] = df["contig_id"].astype(str)
    df["molecule_type"] = df["molecule_type"].str.lower()
    return df[["contig_id", "molecule_type"]].drop_duplicates()


def read_amr(amr_file: Path) -> dict[str, set[str]]:
    skip = 0
    with amr_file.open() as f:
        for line in f:
            if line.startswith("#"):
                skip += 1
            else:
                break
    df = pd.read_csv(amr_file, sep="\t", skiprows=skip, dtype=str)

    contig_col = next(c for c in df.columns if "contig" in c.lower())
    gene_col = next(c for c in df.columns if "gene" in c.lower() or "element" in c.lower())

    out = {}
    for _, r in df.iterrows():
        out.setdefault(r[contig_col], set()).add(r[gene_col])
    return out


def load_kraken_out(path: Path) -> dict[str, dict]:
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    out = {}
    for _, r in df.iterrows():
        contig = str(r[1]).strip()
        raw = str(r[2]).strip()
        taxid = parse_taxid_from_kraken_out_field(raw)
        out[contig] = {"raw": raw, "taxid": taxid}
    return out



def load_kraken_report(path: Path) -> dict[str, str]:
    rep = {}
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) < 6:
                parts = re.split(r"\s+", line.strip(), maxsplit=5)
                if len(parts) < 6:
                    continue

            rank_code = parts[3].strip()
            taxid = parts[4].strip()
            rep[taxid] = rank_code
    return rep



# --------------------------------------------------
# core logic
# --------------------------------------------------
def plasmid_assignment(
    X,
    contig_types,
    amr_map,
    kraken_out,
):
    plasmids = contig_types.query("molecule_type=='plasmid'")["contig_id"]
    plasmids = [p for p in plasmids if p in amr_map and p in X.index]

    chroms = contig_types.query("molecule_type=='chromosome'")["contig_id"]
    chroms = [c for c in chroms if c in X.index]

    rows = []

    for p in plasmids:
        vec_p = X.loc[p].to_numpy(float)
        rmsds = {}

        for c in chroms:
            rv, n = rmsd_overlap(vec_p, X.loc[c].to_numpy(float))
            if n >= MIN_OVERLAP_MOTIFS:
                rmsds[c] = rv

        row = {
            "ctg_id": p,
            "amr_gene_symbols_unique": "; ".join(sorted(amr_map[p])),
            "assignment_method": "nanomotif",
            "assignment_candidates": "NA",
            "n_min_rmsd_ties": "NA",
            "nanomotif_taxon_profile": "NA",
            "final_assignment": "NA",
        }

        if rmsds:
            best = min(rmsds.values())
            ties = [k for k, v in rmsds.items() if abs(v - best) <= ATOL_TIE]

            taxa = [
                clean_taxon(kraken_out.get(t, {}).get("raw", "NA"))
                for t in ties
            ]

            votes = Counter(taxa)
            winner = votes.most_common(1)[0][0]

            row.update({
                "assignment_candidates": "; ".join(ties),
                "n_min_rmsd_ties": len(ties),
                "nanomotif_taxon_profile": "; ".join(
                    f"{k}({v})" for k, v in votes.items()
                ),
                "final_assignment": winner,
            })

        rows.append(row)

    return pd.DataFrame(rows)

def chromosome_assignment(
    X, contig_types, amr_map, kraken_out, kraken_rep
):
    chroms = contig_types.query("molecule_type=='chromosome'")["contig_id"]
    chroms = [c for c in chroms if c in amr_map]

    rows = []
    ref = [c for c in chroms if c in X.index]

    for c in chroms:
        has_vec = c in X.index
        kout = kraken_out.get(c, {})
        taxid = kout.get("taxid", "")
        rank = kraken_rep.get(taxid, "")
        print(f"[DEBUG] Contig: {c}, TaxID: {taxid}, Rank: {rank}")


        row = {
            "ctg_id": c,
            "amr_gene_symbols_unique": "; ".join(sorted(amr_map[c])),
            "has_nanomotif_vector": has_vec,
            "kraken_taxon": kout.get("raw", "NA"),
            "species_association_method": "unassigned",
            "assignment_candidates": "NA",
            "n_min_rmsd_ties": "NA",
            "nanomotif_taxon_profile": "NA",
            "kraken_report_rank_code": rank or "NA",
        }

        # 1) Kraken species-level
        if rank.startswith("S"):
            row["species_association_method"] = "kraken"
            rows.append(row)
            continue

        # 2) Nanomotif RMSD fallback
        if has_vec:
            vec = X.loc[c].to_numpy(float)
            rmsds = {}
            for r in ref:
                if r == c:
                    continue
                rv, n = rmsd_overlap(vec, X.loc[r].to_numpy(float))
                if n >= MIN_OVERLAP_MOTIFS:
                    rmsds[r] = rv

            if rmsds:
                best = min(rmsds.values())
                ties = [k for k, v in rmsds.items() if abs(v - best) <= ATOL_TIE]

                taxa = [clean_taxon(kraken_out.get(t, {}).get("raw", "")) for t in ties]
                votes = Counter(taxa)
                winner = votes.most_common(1)[0][0]

                row.update({
                    "species_association_method": "nanomotif",
                    "assignment_candidates": "; ".join(ties),
                    "n_min_rmsd_ties": len(ties),
                    
                    "nanomotif_taxon_profile": "; ".join(
                        f"{k}({v})" for k, v in votes.items()
                    ),
                })

                row["final_assignment"] = winner

        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------
# main
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nanomotif-dir", type=Path, required=True)
    ap.add_argument("--mobsuite-dir", type=Path, required=True)
    ap.add_argument("--amr-dir", type=Path, required=True)
    ap.add_argument("--kraken-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)

    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    score = find_one(args.nanomotif_dir, ["motifs-scored-read-methylation.tsv"], "Nanomotif score")
    contig_report = find_one(args.mobsuite_dir, ["contig_report.txt"], "MobSuite report")
    amr_file = find_one(args.amr_dir, ["*.tsv", "*.txt"], "AMRFinder output")
    kraken_out_file = find_one(args.kraken_dir, ["*.out", "*kraken*.txt"], "Kraken OUT")
    kraken_report_file = find_one(args.kraken_dir, ["*.report"], "Kraken REPORT")

    X = load_motif_scores(score)
    contig_types = extract_contig_types(contig_report)
    amr_map = read_amr(amr_file)
    kraken_out = load_kraken_out(kraken_out_file)
    kraken_rep = load_kraken_report(kraken_report_file)

    chrom_df = chromosome_assignment(
        X, contig_types, amr_map, kraken_out, kraken_rep
    )

    chrom_df.to_csv(
        args.outdir / "AMR_chromosome_assignment.tsv",
        sep="\t", index=False
    )

    plasmid_df = plasmid_assignment(
        X, contig_types, amr_map, kraken_out
    )
    plasmid_df.to_csv(
        args.outdir / "AMR_plasmid_assignment.tsv",
        sep="\t", index=False
    )
    print("[DONE]")


if __name__ == "__main__":
    main()
