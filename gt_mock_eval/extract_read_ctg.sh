#!/bin/bash

set -euo pipefail

ALIGNED_BAM="path/to/aligned_reads.bam"
MAP_TSV="path/to/output_mapping.tsv"

# Flags:
# -F 2304  = remove secondary (256) + supplementary (2048)
# -F 4     = remove unmapped
samtools view -F 2308 "${ALIGNED_BAM}" \
  | awk 'BEGIN{OFS="\t"} {print $1, $3}' \
  > "${MAP_TSV}"
echo "[02] Done."
echo "  BAM: ${ALIGNED_BAM}"
echo "  Mapping TSV: ${MAP_TSV}"
