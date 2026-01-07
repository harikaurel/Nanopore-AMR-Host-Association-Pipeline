#!/usr/bin/env python3

import os
import glob
import gzip
import pysam

# ============================================================
# CONFIG (edit as needed)
# ============================================================
FASTQ_DIR = "/path/to/chopper_filtered_fastqs"
BAM_DIR   = "/path/to/merged_bams"
OUT_DIR   = "/path/to/filtered_bams"

# FASTQ extensions to consider
FASTQ_GLOBS = ["*.fastq", "*.fq", "*.fastq.gz", "*.fq.gz"]

# If True: require BAM to exist for each FASTQ, otherwise skip
SKIP_IF_NO_BAM = True
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)

def iter_fastqs(folder: str):
    paths = []
    for pat in FASTQ_GLOBS:
        paths.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(set(paths))

def stem_from_fastq(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".gz"):
        base = base[:-3]
    for ext in (".fastq", ".fq"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]

def open_maybe_gz(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")

def get_read_ids_from_fastq(path: str):
    read_ids = set()
    with open_maybe_gz(path) as f:
        for i, line in enumerate(f):
            if i % 4 == 0:  # header line
                header = line.strip()
                rid = header.split()[0].lstrip("@")
                read_ids.add(rid)
    return read_ids

fastq_paths = iter_fastqs(FASTQ_DIR)
if not fastq_paths:
    raise SystemExit(f"No FASTQs found in {FASTQ_DIR}")

print("FASTQs found:")
for fq in fastq_paths:
    print("  -", fq)

for fq in fastq_paths:
    stem = stem_from_fastq(fq)

    bam_path = os.path.join(BAM_DIR, stem + ".bam")
    if not os.path.exists(bam_path):
        msg = f"[NO BAM] FASTQ={fq} expected BAM={bam_path}"
        if SKIP_IF_NO_BAM:
            print(msg + " -> skipping")
            continue
        else:
            raise SystemExit(msg)

    out_bam_path = os.path.join(OUT_DIR, stem + ".filtered.bam")

    print(f"\n=== Processing {stem} ===")
    print(f"FASTQ: {fq}")
    print(f"BAM:   {bam_path}")
    print(f"OUT:   {out_bam_path}")

    read_ids = get_read_ids_from_fastq(fq)
    print(f"  Read IDs in FASTQ: {len(read_ids)}")

    kept = 0
    total = 0

    with pysam.AlignmentFile(bam_path, check_sq=False) as bam_in, \
         pysam.AlignmentFile(out_bam_path, "wb", template=bam_in) as bam_out:

        for aln in bam_in:
            total += 1
            if aln.query_name in read_ids:
                bam_out.write(aln)
                kept += 1

    print(f"  Total alignments in BAM: {total}")
    print(f"  Alignments written: {kept}")

    pysam.index(out_bam_path)
    print(f"  Indexed: {out_bam_path}.bai")

print("\nDone.")
print(f"Filtered BAMs are in: {OUT_DIR}")
