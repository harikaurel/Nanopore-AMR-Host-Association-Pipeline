Perfect — these outputs look **exactly** like what your logic is designed to produce. Below is a **clear, column-by-column explanation** you can reuse in your README, methods, or supplement.

---

## AMR **chromosome** host association output

This table contains **only AMR-carrying chromosome contigs**.

```
ctg_id
amr_gene_symbols_unique
species_association_method
has_nanomotif_vector
n_min_rmsd_ties
nanomotif_taxonomic_association_profile
kraken_taxon
```

### Column explanations

**`ctg_id`**
Contig identifier from the assembly (chromosome contigs only).

---

**`amr_gene_symbols_unique`**
Semicolon-separated list of **unique AMR gene symbols** detected on this chromosome by AMRFinder.
Duplicates are removed.

---

**`species_association_method`**
How the species association was assigned:

* `kraken` → species taken directly from Kraken2
* `nanomotif` → inferred using Nanomotif RMSD (only if Kraken was insufficient)
* `unassigned` → neither Kraken nor Nanomotif could assign

In your table, all rows are `kraken`, meaning **Nanomotif was not needed**.

---

**`has_nanomotif_vector`**
Whether this contig has a Nanomotif methylation vector:

* `TRUE` → contig appears in `motifs-scored-read-methylation.tsv`
* `FALSE` → no Nanomotif information available

Even when `TRUE`, Nanomotif may not be used if Kraken already provides a species.

---

**`n_min_rmsd_ties`**
Number of chromosome contigs tied at the **minimum RMSD**.

* `NA` → RMSD not computed (because Kraken was used)
* Integer → only present when `species_association_method = nanomotif`

---

**`nanomotif_taxonomic_association_profile`**
Taxonomic profile derived from **all minimum-RMSD chromosome matches**.

* `NA` → Nanomotif not used
* Otherwise:
  `Taxon(count); Taxon(count); ...`

---

**`kraken_taxon`**
Taxonomic label taken **directly from Kraken2 output (column 3)**, cleaned of `(taxid ...)`.

Examples:

* `Escherichia coli str. K-12 substr. MG1655`
* `Klebsiella pneumoniae subsp. pneumoniae`

This column is **always filled if Kraken classified the contig**, regardless of assignment method.

---

### Interpretation (chromosomes)

All your AMR chromosomes were confidently classified by Kraken2, so:

* Nanomotif RMSD was **not needed**
* Host association is **taxonomy-driven**
* This is the expected and ideal case

---

## AMR **plasmid** host association output

This table contains **only AMR-carrying plasmid contigs** and their inferred host chromosomes.

```
ctg_id
amr_gene_symbols_unique
n_min_rmsd_ties
nanomotif_taxonomic_association_profile
```

### Column explanations

**`ctg_id`**
Plasmid contig identifier.

---

**`amr_gene_symbols_unique`**
Unique AMR genes detected on the plasmid.

---

**`n_min_rmsd_ties`**
Number of chromosome contigs with the **same minimum RMSD** to this plasmid.

* `1` → a single best host chromosome
* `>1` → ambiguous host signal (multiple equally similar chromosomes)

---

**`nanomotif_taxonomic_association_profile`**
Taxonomic profile of **host chromosomes** that had the minimum RMSD to the plasmid.

Format:

```
Taxon(count)
```

Examples:

* `Escherichia coli(1)`
* `Klebsiella pneumoniae(1)`
* Mixed profiles are possible if multiple ties exist.

This profile is derived **after RMSD**, using Kraken taxonomy of the matched chromosome contigs.