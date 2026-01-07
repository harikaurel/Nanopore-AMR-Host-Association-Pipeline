#!/usr/bin/env python3

from pathlib import Path

# ============================================================
# Generate contig_bin.tsv where each contig is its own bin
# ============================================================

# ------------------------------------------------------------
# Input / output (edit as needed)
# ------------------------------------------------------------
FASTA_FILE = Path("/path/to/assembly.fasta")
OUTPUT_TSV = Path("/path/to/contig_bins.tsv")

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
BIN_PREFIX = "bin_"   # prefix for bin IDs

# ------------------------------------------------------------
# Main logic
# ------------------------------------------------------------
contig_names = []

with FASTA_FILE.open() as f:
    for line in f:
        if line.startswith(">"):
            contig = line[1:].strip().split()[0]
            contig_names.append(contig)

with OUTPUT_TSV.open("w") as out:
    for contig in contig_names:
        out.write(f"{contig}\t{BIN_PREFIX}{contig}\n")

print(f"[INFO] Wrote {len(contig_names)} contig-bin mappings to {OUTPUT_TSV}")
