#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re
from collections import Counter
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


csv.field_size_limit(sys.maxsize)

# ----------------------------
# Defaults
# ----------------------------
MIN_OVERLAP_MOTIFS_DEFAULT = 1
ATOL_TIE_DEFAULT = 0

TAXID_RE = re.compile(r"\(taxid\s+(\d+)\)")


# ----------------------------
# Small utilities
# ----------------------------
def find_one(folder: Path, patterns: list[str], label: str) -> Path:
    hits: list[Path] = []
    for pat in patterns:
        hits.extend(folder.glob(pat))
    hits = [h for h in hits if h.is_file()]
    if not hits:
        raise SystemExit(f"[STOP] Could not find {label} in {folder}")
    if len(hits) > 1:
        print(f"[WARN] Multiple {label} files found, using {hits[0].name}")
    return hits[0]


# ----------------------------
# Kraken2 parsing
# ----------------------------
def parse_taxid_from_kraken_field(field: str) -> str:
    """Extract numeric taxid from strings like 'Bacteroides ... (taxid 1234)'."""
    if not isinstance(field, str):
        return ""
    s = field.strip()
    m = TAXID_RE.search(s)
    if m:
        return m.group(1)
    if s.isdigit():
        return s
    return ""


def clean_taxon(label: str) -> str:
    """Remove '(...)' fragments, collapse whitespace."""
    if not isinstance(label, str):
        return "Unclassified"
    s = label.strip()
    if not s or s.lower() == "nan":
        return "Unclassified"
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else "Unclassified"


def canonical_species_label(taxon: str) -> str:
    """
    Collapse subspecies/strain-ish labels -> 'Genus species' when possible.
    If cannot, return cleaned taxon.
    """
    s = clean_taxon(taxon)
    if s == "Unclassified":
        return s
    # Remove rank-like prefixes if present
    s = re.sub(r"^\s*(s__|g__|p__|k__|c__|o__|f__)\s*", "", s).strip()
    parts = s.split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return s


def load_kraken_out(path: Path) -> dict[str, dict]:
    """
    Kraken OUT (tab-separated), typically:
      col0: C/U
      col1: contig/read id
      col2: taxon label (often includes '(taxid N)')
      ...
    Returns: contig -> {raw, taxid, species_can}
    """
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if df.shape[1] < 3:
        raise ValueError(f"Kraken OUT has <3 columns: {path}")

    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        contig = str(r.iloc[1]).strip()
        raw = str(r.iloc[2]).strip() if pd.notna(r.iloc[2]) else "Unclassified"
        taxid = parse_taxid_from_kraken_field(raw)
        out[contig] = {
            "raw": raw if raw else "Unclassified",
            "taxid": taxid,
            "species_can": canonical_species_label(raw),
        }
    return out


def load_kraken_report_rank_map(path: Path) -> dict[str, str]:
    """
    Kraken report lines usually contain:
      percent, clade_reads, taxon_reads, rank_code, taxid, name
    Returns: taxid -> rank_code (S/G/F/...)
    """
    rep: dict[str, str] = {}
    with path.open("r", errors="replace") as f:
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
            if taxid:
                rep[taxid] = rank_code
    return rep


# ----------------------------
# AMRFinder parsing
# ----------------------------
def read_table_skip_hash(path: Path) -> pd.DataFrame:
    skip = 0
    with path.open("r", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                skip += 1
            else:
                break
    return pd.read_csv(path, sep="\t", skiprows=skip, dtype=str)


def read_amr_map(amr_file: Path) -> dict[str, set[str]]:
    df = read_table_skip_hash(amr_file)
    contig_col = next(c for c in df.columns if "contig" in c.lower())
    gene_col = next(c for c in df.columns if ("element" in c.lower()) or ("gene" in c.lower()))

    out: dict[str, set[str]] = {}
    for _, r in df.iterrows():
        ctg = str(r[contig_col]).strip()
        gene = str(r[gene_col]).strip()
        if ctg and gene and ctg.lower() != "nan":
            out.setdefault(ctg, set()).add(gene)
    return out


# ----------------------------
# MobSuite contig types
# ----------------------------
def extract_contig_types(contig_report: Path) -> pd.DataFrame:
    rep = pd.read_csv(contig_report, sep="\t", dtype=str)
    rep["contig_id"] = rep["contig_id"].astype(str).str.strip()
    rep["molecule_type"] = rep["molecule_type"].astype(str).str.lower().str.strip()
    return rep[["contig_id", "molecule_type"]].drop_duplicates()


# ----------------------------
# Nanomotif loaders and fast arrays
# ----------------------------
def load_motif_scores_long(path: Path) -> pd.DataFrame:
    """
    Reads motifs-scored-read-methylation.tsv and collapses to per-(contig,motif) median(median).
    Must contain columns: contig, motif, median
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "median" not in df.columns:
        raise ValueError(f"'median' column not found in {path}. Columns: {list(df.columns)}")

    df["median"] = pd.to_numeric(df["median"], errors="coerce")
    df["contig"] = df["contig"].astype(str).str.strip()
    df["motif"] = df["motif"].astype(str).str.strip()

    df = df.groupby(["contig", "motif"], as_index=False)["median"].median()
    df = df.dropna(subset=["median"])
    return df


def build_contig_arrays(df_long: pd.DataFrame):
    """
    Build fast representation:
      contig2motifs[contig] = sorted array of motif strings
      contig2vals[contig]   = aligned array of medians
    """
    contig2motifs: dict[str, np.ndarray] = {}
    contig2vals: dict[str, np.ndarray] = {}

    for contig, sub in df_long.groupby("contig", sort=False):
        m = sub["motif"].to_numpy(dtype=object)
        v = sub["median"].to_numpy(dtype=float)
        order = np.argsort(m)
        contig2motifs[contig] = m[order]
        contig2vals[contig] = v[order]

    return contig2motifs, contig2vals


def rmsd_nm_shared(contigA: str, contigB: str, contig2motifs, contig2vals) -> tuple[float, int]:
    ma = contig2motifs.get(contigA)
    mb = contig2motifs.get(contigB)
    if ma is None or mb is None:
        return np.nan, 0

    shared, ia, ib = np.intersect1d(ma, mb, assume_unique=False, return_indices=True)
    nm = int(shared.size)
    if nm == 0:
        return np.nan, 0

    va = contig2vals[contigA][ia]
    vb = contig2vals[contigB][ib]
    rmsd = float(np.sqrt(np.mean((va - vb) ** 2)))
    return rmsd, nm


def score_candidates(
    query_id: str,
    candidate_ids: list[str],
    contig2motifs,
    contig2vals,
    kraken_out: dict[str, dict],
    min_overlap: int,
):
    """
    Candidate table ranked by final_score.

    Keeps Unclassified candidates (per your requirement).
    """
    rows = []
    for c in candidate_ids:
        if c == query_id:
            continue

        rmsd, nm = rmsd_nm_shared(query_id, c, contig2motifs, contig2vals)
        if nm < min_overlap or not np.isfinite(rmsd):
            continue

        rmss = max(0.0, 1.0 - rmsd)
        final_score = rmss * nm

        raw = kraken_out.get(c, {}).get("raw", "Unclassified")
        species_can = canonical_species_label(raw)  # may be "Unclassified" and that's OK

        rows.append((c, rmsd, nm, rmss, final_score, species_can))

    df = pd.DataFrame(rows, columns=["candidate", "rmsd", "shared_motifs", "rmss", "final_score", "species"])
    if df.empty:
        return df

    df = df.sort_values(["final_score", "shared_motifs", "rmsd"], ascending=[False, False, True]).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def tie_species_summary_one_line(df_at_max: pd.DataFrame) -> str:
    """Return 'SpeciesA (n); SpeciesB (m); ...' from df_at_max['species']."""
    counts = df_at_max["species"].astype(str).value_counts()
    return "; ".join([f"{sp} ({int(n)})" for sp, n in counts.items()])


def pick_winner_species_from_top_ties(top_df: pd.DataFrame, prefer_non_unclassified: bool = True) -> str:
    """
    Choose winner species among top-score tied candidates by counts.
    If prefer_non_unclassified=True and there is at least one non-Unclassified,
    ignore Unclassified only for choosing the winner.
    """
    counts = Counter(top_df["species"].astype(str).tolist())
    if prefer_non_unclassified:
        if any(sp != "Unclassified" for sp in counts):
            counts.pop("Unclassified", None)
    return counts.most_common(1)[0][0] if counts else "Unclassified"

# ----------------------------
# plot
# ----------------------------
def plot_candidate_dotplot(
    cand_df: pd.DataFrame,
    query_id: str,
    out_png: Path,
    kraken_out: dict[str, dict],
    atol_tie: float = 0.0,
    max_legend_items: int = 12,
):
    """
    Dot plot:
      x-axis: ranked chromosome contigs (cand_df['rank'])
      y-axis: contig similarity score (cand_df['final_score'])
    Only highlight (color) the top-score candidate(s) (ties within atol_tie),
    and add their Kraken2 taxon labels to the legend.
    """
    if cand_df is None or cand_df.empty:
        return

    df = cand_df.copy()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")
    df = df.dropna(subset=["rank", "final_score"])
    if df.empty:
        return

    max_score = float(df["final_score"].max())
    is_top = df["final_score"] >= (max_score - float(atol_tie))
    top_df = df.loc[is_top].copy()
    rest_df = df.loc[~is_top].copy()

    plt.figure(figsize=(10, 4.5))

    # Background points (all non-top)
    if not rest_df.empty:
        plt.scatter(
            rest_df["rank"].to_numpy(),
            rest_df["final_score"].to_numpy(),
            s=18,
            c="0.75",       # light gray
            alpha=0.8,
            linewidths=0,
            label=None,
        )

    # Highlight top points (ties), one-by-one so each can have its own legend label
    if not top_df.empty:
        # Keep legend readable if there are tons of ties
        top_df = top_df.sort_values(["final_score", "shared_motifs", "rmsd"], ascending=[False, False, True])
        if len(top_df) > max_legend_items:
            top_df = top_df.head(max_legend_items)

        for _, r in top_df.iterrows():
            cand = str(r["candidate"])
            raw = kraken_out.get(cand, {}).get("raw", "Unclassified")
            taxon = clean_taxon(raw)

            # Legend label requested: include Kraken2 taxon
            # (also include candidate id so duplicates are clearer)
            lab = f"{taxon} — {cand}"

            plt.scatter(
                [float(r["rank"])],
                [float(r["final_score"])],
                s=48,
                alpha=0.95,
                linewidths=0.5,
                edgecolors="black",
                label=lab,
            )

    plt.xlabel("Ranked chromosome contigs")
    plt.ylabel("Contig similarity score")
    plt.title(f"{query_id}")

    # Only show legend if we have highlighted points
    if not top_df.empty:
        plt.legend(
            loc="best",
            frameon=False,
            fontsize=8,
            handlelength=1.2,
            borderaxespad=0.5,
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ----------------------------
# Core pipeline
# ----------------------------
def run_assignment(
    contig_types: pd.DataFrame,
    amr_map: dict[str, set[str]],
    kraken_out: dict[str, dict],
    kraken_rank: dict[str, str],
    contig2motifs,
    contig2vals,
    outdir: Path,
    min_overlap_motifs: int,
    atol_tie: float,
    prefer_non_unclassified_vote: bool,
):
    outdir.mkdir(parents=True, exist_ok=True)

    # Candidate/reference pool: ALL chromosomes with Nanomotif vectors (NOT limited to AMR)
    chroms_all = contig_types.query("molecule_type == 'chromosome'")["contig_id"].astype(str).tolist()
    chroms_vec = [c for c in chroms_all if c in contig2motifs]

    plasmids_all = contig_types.query("molecule_type == 'plasmid'")["contig_id"].astype(str).tolist()
    chromosomes_all = contig_types.query("molecule_type == 'chromosome'")["contig_id"].astype(str).tolist()

    def base_row(ctg: str, mol_type: str) -> dict:
        kout = kraken_out.get(ctg, {})
        taxid = kout.get("taxid", "")
        rank_code = kraken_rank.get(taxid, "")
        raw = kout.get("raw", "Unclassified")
        return {
            "contig_id": ctg,
            "molecule_type": mol_type,
            "amr_genes": "; ".join(sorted(amr_map.get(ctg, set()))),
            "has_nanomotif_vector": (ctg in contig2motifs),

            "kraken_raw": raw,
            # "kraken_taxid": taxid or "NA",
            # "kraken_rank_code": rank_code or "NA",
            # "kraken_species_can": canonical_species_label(raw),

            "assignment_method": "unassigned",
            "best_species": "NA",
            "best_candidate": "NA",
            "best_final_score": np.nan,
            # "best_rmsd": np.nan,
            # "best_shared_motifs": 0,
            # "n_top_score_candidates": 0,
            "top_score_species_counts": "NA",
        }

    plasmid_rows = []
    chrom_rows = []

    # ----------------------------
    # PLASMIDS (queries = AMR+ plasmids)
    # ----------------------------
    for p in plasmids_all:
        if p not in amr_map:
            continue  # report only AMR+ queries
        row = base_row(p, "plasmid")

        if p not in contig2motifs:
            row["assignment_method"] = "no_nanomotif_vector"
            plasmid_rows.append(row)
            continue

        cand_df = score_candidates(
            query_id=p,
            candidate_ids=chroms_vec,  # all chromosome vectors
            contig2motifs=contig2motifs,
            contig2vals=contig2vals,
            kraken_out=kraken_out,
            min_overlap=min_overlap_motifs,
        )
                # Dot plot (rank vs final_score), highlight top score ties
        plot_candidate_dotplot(
            cand_df=cand_df,
            query_id=p,
            out_png=outdir / "plots" / f"{p}.dotplot.png",
            kraken_out=kraken_out,
            atol_tie=atol_tie,
        )

        if cand_df.empty:
            row["assignment_method"] = "nanomotif_no_candidates"
            plasmid_rows.append(row)
            continue

        max_score = float(cand_df["final_score"].max())
        at_max = cand_df["final_score"] >= (max_score - atol_tie)
        top_df = cand_df.loc[at_max].copy()

        best_species = pick_winner_species_from_top_ties(top_df, prefer_non_unclassified_vote)
        best_row = cand_df.iloc[0]

        row.update({
            "assignment_method": "nanomotif_final_score",
            "best_species": best_species,
            "best_candidate": str(best_row["candidate"]),
            "best_final_score": float(best_row["final_score"]),
            # "best_rmsd": float(best_row["rmsd"]),
            # "best_shared_motifs": int(best_row["shared_motifs"]),
            # "n_top_score_candidates": int(len(top_df)),
            "top_score_species_counts": tie_species_summary_one_line(top_df),
        })

        # cand_df.head(50).to_csv(outdir / f"{p}.top50_candidates.tsv", sep="\t", index=False)
        plasmid_rows.append(row)

    # ----------------------------
    # CHROMOSOMES (queries = AMR+ chromosomes)
    # ----------------------------
    for c in chromosomes_all:
        if c not in amr_map:
            continue  # report only AMR+ queries
        row = base_row(c, "chromosome")

        kout = kraken_out.get(c, {})
        taxid = kout.get("taxid", "")
        rank_code = kraken_rank.get(taxid, "")

        # (1) If Kraken says species-level => accept Kraken directly
        if isinstance(rank_code, str) and rank_code.startswith("S"):
            row["assignment_method"] = "kraken_species_level"
            row["best_species"] = kout.get("species_can", canonical_species_label(kout.get("raw", "Unclassified")))
            chrom_rows.append(row)
            continue

        # (2) Not species-level: do Nanomotif fallback (if vector exists)
        if c not in contig2motifs:
            row["assignment_method"] = "non_species_kraken_no_nanomotif_vector"
            chrom_rows.append(row)
            continue

        cand_ids = [x for x in chroms_vec if x != c]  # exclude itself
        cand_df = score_candidates(
            query_id=c,
            candidate_ids=cand_ids,
            contig2motifs=contig2motifs,
            contig2vals=contig2vals,
            kraken_out=kraken_out,
            min_overlap=min_overlap_motifs,
        )

        plot_candidate_dotplot(
            cand_df=cand_df,
            query_id=c,
            out_png=outdir / "plots" / f"{c}.dotplot.png",
            kraken_out=kraken_out,
            atol_tie=atol_tie,
        )

        if cand_df.empty:
            row["assignment_method"] = "nanomotif_no_candidates"
            chrom_rows.append(row)
            continue

        max_score = float(cand_df["final_score"].max())
        at_max = cand_df["final_score"] >= (max_score - atol_tie)
        top_df = cand_df.loc[at_max].copy()

        best_species = pick_winner_species_from_top_ties(top_df, prefer_non_unclassified_vote)
        best_row = cand_df.iloc[0]

        row.update({
            "assignment_method": "nanomotif_final_score_non_species_kraken",
            "best_species": best_species,
            "best_candidate": str(best_row["candidate"]),
            "best_final_score": float(best_row["final_score"]),
            # "best_rmsd": float(best_row["rmsd"]),
            # "best_shared_motifs": int(best_row["shared_motifs"]),
            # "n_top_score_candidates": int(len(top_df)),
            "top_score_species_counts": tie_species_summary_one_line(top_df),
        })

        # cand_df.head(50).to_csv(outdir / f"{c}.top50_candidates.tsv", sep="\t", index=False)
        chrom_rows.append(row)

    plasmid_df = pd.DataFrame(plasmid_rows)
    chrom_df = pd.DataFrame(chrom_rows)

    plasmid_df.to_csv(outdir / "AMR_plasmid_assignment.tsv", sep="\t", index=False)
    chrom_df.to_csv(outdir / "AMR_chromosome_assignment.tsv", sep="\t", index=False)

    # if not plasmid_df.empty or not chrom_df.empty:
    #     combined = pd.concat([plasmid_df, chrom_df], ignore_index=True)
    #     combined.to_csv(outdir / "AMR_all_assignment.tsv", sep="\t", index=False)

    print("[DONE] Outputs written to:", str(outdir))
    print("  - AMR_plasmid_assignment.tsv")
    print("  - AMR_chromosome_assignment.tsv")
    print("  - AMR_all_assignment.tsv")
    print("  - <contig>.top50_candidates.tsv (per query)")


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="AMR contig association using Nanomotif final_score=(max(0,1-RMSD))*NM and Kraken2 species-level override."
    )
    ap.add_argument("--nanomotif-dir", type=Path, required=True, help="Folder containing motifs-scored-read-methylation.tsv")
    ap.add_argument("--mobsuite-dir", type=Path, required=True, help="Folder containing contig_report.txt")
    ap.add_argument("--amr-dir", type=Path, required=True, help="Folder containing AMRFinder output (.tsv/.txt)")
    ap.add_argument("--kraken-dir", type=Path, required=True, help="Folder containing kraken .out and .report")
    ap.add_argument("--outdir", type=Path, required=True, help="Output directory")

    ap.add_argument(
        "--prefer-non-unclassified-vote",
        action="store_true",
        help="If top-score ties include real species, ignore Unclassified only for choosing the winner species.",
    )

    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    score = find_one(args.nanomotif_dir, ["motifs-scored-read-methylation.tsv"], "Nanomotif score TSV")
    contig_report = find_one(args.mobsuite_dir, ["contig_report.txt"], "MobSuite contig_report.txt")
    amr_file = find_one(args.amr_dir, ["*.tsv", "*.txt"], "AMRFinder output")
    kraken_out_file = find_one(args.kraken_dir, ["*.out", "*kraken*.out", "*kraken*.txt"], "Kraken OUT")
    kraken_report_file = find_one(args.kraken_dir, ["*.report", "*kraken*.report"], "Kraken REPORT")

    print("[1/6] Loading Nanomotif long table...")
    df_long = load_motif_scores_long(score)

    print("[2/6] Building fast contig arrays...")
    contig2motifs, contig2vals = build_contig_arrays(df_long)

    print("[3/6] Loading contig types (MobSuite)...")
    contig_types = extract_contig_types(contig_report)

    print("[4/6] Loading AMR map (AMRFinder)...")
    amr_map = read_amr_map(amr_file)

    print("[5/6] Loading Kraken OUT + REPORT rank map...")
    kraken_out = load_kraken_out(kraken_out_file)
    kraken_rank = load_kraken_report_rank_map(kraken_report_file)

    print("[6/6] Running assignment...")
    run_assignment(
        contig_types=contig_types,
        amr_map=amr_map,
        kraken_out=kraken_out,
        kraken_rank=kraken_rank,
        contig2motifs=contig2motifs,
        contig2vals=contig2vals,
        outdir=args.outdir,
        min_overlap_motifs=int(MIN_OVERLAP_MOTIFS_DEFAULT),
        atol_tie=float(ATOL_TIE_DEFAULT),
        prefer_non_unclassified_vote=bool(args.prefer_non_unclassified_vote),
    )


if __name__ == "__main__":
    main()
