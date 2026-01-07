#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Nanomotif: motif discovery + motif–contig scoring
# (detect_contamination is used only to produce the motif–contig score file)
# ============================================================

NANOMOTIF_BIN="/path/to/nanomotif"

ASSEMBLY_FASTA="/path/to/assembly.fasta"
PILEUP_BED="/path/to/pileup.bed"
CONTIG_BIN_TSV="/path/to/contig2bin.tsv"

OUT_DIR="/path/to/nanomotif_output"

THREADS=20
VALID_COVERAGE_THRESHOLD=10

mkdir -p "$OUT_DIR"

# 1) Motif discovery
"$NANOMOTIF_BIN" motif_discovery \
  "$ASSEMBLY_FASTA" \
  "$PILEUP_BED" \
  -c "$CONTIG_BIN_TSV" \
  --out "$OUT_DIR" \
  -t "$THREADS" \
  --threshold_valid_coverage "$VALID_COVERAGE_THRESHOLD"

# 2) Motif–contig scoring (via detect_contamination)
BIN_MOTIFS_TSV="${OUT_DIR}/bin-motifs.tsv"   # produced by motif_discovery

"$NANOMOTIF_BIN" detect_contamination \
  --pileup "$PILEUP_BED" \
  --assembly "$ASSEMBLY_FASTA" \
  --bin_motifs "$BIN_MOTIFS_TSV" \
  --contig_bins "$CONTIG_BIN_TSV" \
  --out "$OUT_DIR"
