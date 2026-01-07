#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Modkit pileup
# ============================================================

# ------------------------------------------------------------
# Tool paths (edit as needed)
# ------------------------------------------------------------
MODKIT_BIN="/path/to/modkit"
SAMTOOLS_BIN="/path/to/samtools"

# ------------------------------------------------------------
# Input / output
# ------------------------------------------------------------
REF_FASTA="/path/to/reference.fasta"
ALN_BAM="/path/to/aligned.sorted.bam"
OUT_DIR="/path/to/modkit_pileup_out"
PREFIX="sample"   # output prefix

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
THREADS=20

# Optional: set to 1 if you want to force BAM indexing here
ENSURE_INDEX=1

# ------------------------------------------------------------
# Prepare output directory
# ------------------------------------------------------------
mkdir -p "$OUT_DIR"

# ------------------------------------------------------------
# Ensure BAM index
# ------------------------------------------------------------
if [[ "$ENSURE_INDEX" == "1" ]]; then
  "$SAMTOOLS_BIN" index -@ "$THREADS" "$ALN_BAM"
fi

# ------------------------------------------------------------
# Run pileup
# Output is typically a bedMethyl-like file (often .bed / .bed.gz) depending on args.
# ------------------------------------------------------------
"$MODKIT_BIN" pileup \
  --ref "$REF_FASTA" \
  --threads "$THREADS" \
  "$ALN_BAM" \
  "$OUT_DIR/${PREFIX}.pileup.bed"

