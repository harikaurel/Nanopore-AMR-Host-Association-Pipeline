#!/usr/bin/env bash

set -euo pipefail

BAM="${1:?usage: $0 <in.bam> [out.tsv] [min_mapq]}"
MINQ="${3:-0}"
OUT="${2:-${BAM%.bam}_contig_coverage.tsv}"

[[ -f "$BAM" ]] || { echo "ERROR: BAM not found: $BAM" >&2; exit 1; }
[[ -f "${BAM}.bai" || -f "${BAM%.bam}.bai" ]] || samtools index "$BAM"

samtools coverage --ff UNMAP,SECONDARY,SUPPLEMENTARY,QCFAIL,DUP -q "$MINQ" "$BAM" \
  | awk 'BEGIN{FS=OFS="\t"; print "contig","coverage","length","breadth","meanmapq"}
         NR>1 {print $1, $7, $3-$2+1, $6, $9}' \
  > "$OUT"

echo "wrote $OUT" >&2