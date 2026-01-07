#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AMR-only host association for ONE unit.

You provide folders for:
- Nanomotif
- MobSuite
- AMRFinder
- Kraken

"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re
from collections import Counter
import csv
import sys

csv.field_size_limit(sys.maxsize)

# --------------------------------------------------
# helpers
# --------------------------------------------------
def find_one(folder: Path, patterns: list[str], label: str) -> Path:
    hits = []
    for pat in patterns:
        hits.extend(folder.glob(pat))
    hits = [h for h in hits if h.is_file()]

    if not hits:
        raise SystemExit(f"[STOP] Could not find {label} in {folder}")
    if len(hits) > 1:
        print(f"[WARN] Multiple {label} files found in {folder}, using {hits[0].name}")
    return hits[0]

def clean_taxon(label: str) -> str:
    if not isinstance(label, str) or not label.strip():
        return ""
    s = re.sub(r"\([^)]*\)", "", label).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def parse_species(label: str) -> str:
    s = clean_taxon(label)
    if not s:
        return "Unclassified"
    toks = s.split()
    return " ".join(toks[:2]) if len(toks) >= 2 else toks[0]

def kraken_to_species_or_root(label: str) -> str:
    if not isinstance(label, str) or not label.strip():
        return "root"
    s = re.sub(r"\([^)]*\)", "", label).strip()
    s = re.sub(r"\s+", " ", s)
    toks = s.split()
    if len(toks) >= 2:
        return f"{toks[0]} {toks[1]}"
    return "root"


# --------------------------------------------------
# loaders
# --------------------------------------------------
def load_motif_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    print(f"[INFO] Loaded motif scores for {df['contig'].nunique()} contigs and {df['motif'].nunique()} motifs")
    df["median"] = df["median"].astype(float)
    df = df.groupby(["contig", "motif"], as_index=False)["median"].median()
    X = df.pivot(index="contig", columns="motif", values="median").fillna(0.0)
    X.index = X.index.astype(str).str.strip()
    return X


def extract_contig_types(contig_report: Path) -> pd.DataFrame:
    rep = pd.read_csv(contig_report, sep="\t", dtype=str)
    print(f"[INFO] Loaded MobSuite contig report with {rep.shape[0]} contigs")
    rep["contig_id"] = rep["contig_id"].astype(str).str.strip()
    rep["molecule_type"] = rep["molecule_type"].astype(str).str.lower().str.strip()
    return rep[["contig_id", "molecule_type"]].drop_duplicates()


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
    gene_col = next(c for c in df.columns if "element" in c.lower() or "gene" in c.lower())
    print(f"[INFO] Loaded AMR data with {df.shape[0]} entries from {amr_file.name}")

    out = {}
    for _, r in df.iterrows():
        ctg = str(r[contig_col]).strip()
        gene = str(r[gene_col]).strip()
        if ctg and gene:
            out.setdefault(ctg, set()).add(gene)
            # print(f"[DEBUG] AMR gene '{gene}' found on contig '{ctg}'")
    return out


def load_kraken(kraken_file: Path) -> dict[str, str]:
    df = pd.read_csv(kraken_file, sep="\t", header=None, dtype=str)
    print(f"[INFO] Loaded Kraken classifications for {df[1].nunique()} contigs from {kraken_file.name}")
    return dict(zip(df[1].astype(str).str.strip(), df[2].astype(str).str.strip()))


def parse_species(label: str) -> str:
    if not isinstance(label, str):
        return "Unclassified"
    s = re.sub(r"\([^)]*\)", "", label).strip()
    toks = s.split()
    return " ".join(toks[:2]) if len(toks) >= 2 else toks[0]

def load_kraken_taxon(label: str) -> str:
    # take Kraken column 3 content and just clean "(taxid ...)" part
    if not isinstance(label, str) or not label.strip():
        return "NA"
    s = re.sub(r"\([^)]*\)", "", label).strip()
    s = re.sub(r"\s+", " ", s)
    return s if s else "NA"

def is_binomial_species(taxon: str) -> bool:
    if not isinstance(taxon, str):
        return False
    toks = taxon.split()
    if len(toks) < 2:
        return False
    # reject placeholders
    if toks[1].lower() in {"sp", "sp.", "spp", "spp.", "bacterium"}:
        return False
    return True


# --------------------------------------------------
# computations
# --------------------------------------------------
def plasmid_host_association(X, contig_types, amr_map, kraken):
    plasmids = contig_types.query("molecule_type=='plasmid'")["contig_id"].astype(str).tolist()
    chroms   = contig_types.query("molecule_type=='chromosome'")["contig_id"].astype(str).tolist()

    # AMR-only plasmids, and require Nanomotif vector
    plasmids = [p for p in plasmids if p in amr_map and p in X.index]
    chroms   = [c for c in chroms if c in X.index]

    rows = []
    if not plasmids or not chroms:
        return pd.DataFrame(rows)

    C = X.loc[chroms].to_numpy(float)

    for p in plasmids:
        P = X.loc[p].to_numpy(float)
        rmsd = np.sqrt(np.mean((C - P) ** 2, axis=1))
        rmin = float(rmsd.min())

        # exact min ties only (same as your current behavior)
        tie_idx = np.where(rmsd == rmin)[0]
        ties = [chroms[i] for i in tie_idx]

        # fallback (just in case)
        if not ties:
            ties = [chroms[int(np.argmin(rmsd))]]

        # Use FULL cleaned taxon labels for the profile
        taxa = [clean_taxon(kraken.get(c, "Unclassified")) for c in ties]
        prof = "; ".join(f"{k}({v})" for k, v in Counter(taxa).most_common())

        rows.append({
            "ctg_id": p,
            "amr_gene_symbols_unique": "; ".join(sorted(amr_map.get(p, set()))),
            "n_min_rmsd_ties": int(len(ties)),
            "nanomotif_taxonomic_association_profile": prof,
            "assigned_chromosome_contig_ids": "; ".join(ties)
        })

    return pd.DataFrame(rows)


def chromosome_assignment(X, contig_types, amr_map, kraken):
    chroms = contig_types.query("molecule_type=='chromosome'")["contig_id"].astype(str).tolist()
    chroms = [c for c in chroms if c in amr_map]  # AMR-only

    rows = []
    ref = [c for c in chroms if c in X.index]

    for c in chroms:
        has_vec = c in X.index

        raw_label = kraken.get(c, "")  # this is Kraken col3 in your loaded map
        kraken_taxon = load_kraken_taxon(raw_label)

        row = {
            "ctg_id": c,
            "amr_gene_symbols_unique": "; ".join(sorted(amr_map.get(c, set()))),
            "species_association_method": "unassigned",
            "has_nanomotif_vector": "TRUE" if has_vec else "FALSE",
            "n_min_rmsd_ties": "NA",
            "nanomotif_taxonomic_association_profile": "NA",
            "kraken_taxon": kraken_taxon,
        }

        # 1) Kraken species-level -> use kraken
        if kraken_taxon != "NA" and is_binomial_species(kraken_taxon):
            row["species_association_method"] = "kraken"
            rows.append(row)
            continue

        # 2) Kraken not species-level -> try nanomotif
        if has_vec and ref:
            refs = [r for r in ref if r != c]
            if refs:
                vec = X.loc[c].to_numpy(float)
                R = X.loc[refs].to_numpy(float)

                rmsd = np.sqrt(np.mean((R - vec) ** 2, axis=1))
                rmin = float(rmsd.min())

                ties = [refs[i] for i in np.where(rmsd == rmin)[0]]
                if not ties:
                    ties = [refs[int(np.argmin(rmsd))]]

                taxa = [load_kraken_taxon(kraken.get(t, "")) for t in ties]
                prof = "; ".join(f"{k}({v})" for k, v in Counter(taxa).most_common())

                row["species_association_method"] = "nanomotif"
                row["n_min_rmsd_ties"] = str(len(ties))
                row["nanomotif_taxonomic_association_profile"] = prof if prof else "NA"
                row["assigned_chromosome_contig_ids"] = "; ".join(ties)

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

    unit = args.nanomotif_dir.name
    args.outdir.mkdir(parents=True, exist_ok=True)

    score = find_one(args.nanomotif_dir, ["motifs-scored-read-methylation.tsv"], "Nanomotif score")
    contig_report = find_one(args.mobsuite_dir, ["contig_report.txt"], "MobSuite contig_report")
    amr_file = find_one(args.amr_dir, ["*.txt", "*.tsv"], "AMRFinder output")

    kraken = {}
    if args.kraken_dir is not None and str(args.kraken_dir).lower() != "none":
        kraken_file = find_one(args.kraken_dir, ["out.txt", "*.out", "*kraken*.txt"], "Kraken output")
        kraken = load_kraken(kraken_file)

    X = load_motif_scores(score)
    contig_types = extract_contig_types(contig_report)
    amr_map = read_amr(amr_file)

    plasmid_df = plasmid_host_association(X, contig_types, amr_map, kraken)
    chrom_df = chromosome_assignment(X, contig_types, amr_map, kraken)

    if not plasmid_df.empty:
        plasmid_df.to_csv(args.outdir / f"{unit}_AMR_plasmid_host_association.tsv",
                           sep="\t", index=False)

    if not chrom_df.empty:
        chrom_df.to_csv(args.outdir / f"{unit}_AMR_chromosome_assignment.tsv",
                         sep="\t", index=False)

    print("[DONE]")


if __name__ == "__main__":
    main()
