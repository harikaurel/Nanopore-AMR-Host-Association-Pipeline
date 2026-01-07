#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Dorado aligner -> sorted + indexed BAM
# ============================================================

# ------------------------------------------------------------
# Tool paths (edit as needed)
# ------------------------------------------------------------
DORADO_BIN="/path/to/dorado/bin/dorado"
SAMTOOLS_BIN="/path/to/samtools"

# ------------------------------------------------------------
# Input / output
# ------------------------------------------------------------
REF_FASTA="/path/to/reference.fasta"
READS_BAM="/path/to/reads.bam"
ALN_BAM="/path/to/aligned.sorted.bam"

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
THREADS=20

# ------------------------------------------------------------
# Prepare output directory
# ------------------------------------------------------------
mkdir -p "$(dirname "$ALN_BAM")"

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
"$DORADO_BIN" aligner "$REF_FASTA" "$READS_BAM" \
  | "$SAMTOOLS_BIN" sort -@ "$THREADS" -o "$ALN_BAM"

"$SAMTOOLS_BIN" index -@ "$THREADS" "$ALN_BAM"
