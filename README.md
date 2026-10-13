# crDRP-Bench

Materials for **crDRP-Bench: a controlled benchmark of cell-line-trained breast cancer drug-response prediction under tumor admixture**. This snapshot was prepared on 13 October 2026 from the REV7 manuscript materials.

The current supplementary tables and all 24 figures are included. The available source code is preserved under [`legacy/`](legacy/README.md), with its original analysis dates. **The corrected training pipeline, revision-specific inputs, out-of-fold predictions and model weights are still missing from this snapshot.** See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) before attempting to reproduce the revised results.

## Contents

| Location | Contents |
| --- | --- |
| [`data/Supplementary_Tables_S1-S15_REV7.xlsx`](data/Supplementary_Tables_S1-S15_REV7.xlsx) | Original REV7 workbook, including all 63 worksheets |
| [`out/revision/tables/`](out/revision/tables/) | 62 CSV tables exported from the workbook, plus an index with descriptions and dimensions |
| [`figures/main/`](figures/main/) | Figures 1–6, PDF and PNG |
| [`figures/supplementary/`](figures/supplementary/) | Figures S1–S18, PDF and PNG |
| [`docs/FIGURE_CAPTIONS.md`](docs/FIGURE_CAPTIONS.md) | Captions from the clean REV7 documents |
| [`docs/Supplementary_Information_REV7.txt`](docs/Supplementary_Information_REV7.txt) | Text, methods and references extracted from the REV7 supplementary information |
| [`docs/TABLE_INDEX.md`](docs/TABLE_INDEX.md) | Worksheet descriptions and links to individual CSV tables |
| [`DATA_SOURCES.md`](DATA_SOURCES.md) | Source accessions and releases reported in REV7 |
| [`legacy/`](legacy/README.md) | Original August code and local September revision analyses |
| [`scripts/revision/README.md`](scripts/revision/README.md) | Status of the missing corrected revision scripts |
| [`scripts/figure_annotations/`](scripts/figure_annotations/README.md) | Original REV7 vector text/layout-edit source and edit log |
| [`tools/`](tools/) | Table-export and package-integrity utilities |
| [`release_status.json`](release_status.json) | Machine-readable availability and completeness status |
| [`manifest.sha256`](manifest.sha256) | SHA-256 checksums for this repository snapshot |

## Reading the current results

The workbook is the authoritative numerical source. Its title and description rows are retained there. Each CSV starts at the table header, followed by the original data rows; descriptions are recorded in [`table_index.csv`](out/revision/tables/table_index.csv). Formatting, merged cells and cell notes remain available in the XLSX file.

Key tables include:

- [`S1a_master.csv`](out/revision/tables/S1a_master.csv): benchmark summaries, with historical audit rows explicitly labeled in the source.
- [`S1e_contrasts_seed_aware.csv`](out/revision/tables/S1e_contrasts_seed_aware.csv): paired contrasts, bootstrap intervals and reported t-widened sensitivity ranges.
- [`S3e_identity_audit.csv`](out/revision/tables/S3e_identity_audit.csv) and [`S3f_corrections.csv`](out/revision/tables/S3f_corrections.csv): compound-identity audit and corrections.
- [`S15a_qc_flow.csv`](out/revision/tables/S15a_qc_flow.csv) through [`S15d_per_donor.csv`](out/revision/tables/S15d_per_donor.csv): single-cell reference audit summaries.
- [`RV_compare_v1_v2.csv`](out/revision/tables/RV_compare_v1_v2.csv): values before and after the structure correction.

The corrected drug-feature library contains 1,475 compounds. The modeled set contains 117 drugs: 116 with verified or corrected structures, plus BEN with a neutral representation. The reported pair-holdout median paired contrast is +0.000 for additive full minus bulk, with a t-widened range of −0.055 to 0.040. The Transformer contrast is +0.087, with a range of 0.028 to 0.141. Projecting bulk expression into the malignant basis reproduces the Transformer gain: full minus that control is +0.001, with a range of −0.030 to 0.034. These are reported REV7 summaries; the underlying corrected predictions are not included here.

Some supplied tables and figure panels deliberately retain superseded results for provenance. Table S1b, Table S1c, historical audit rows, and Figures S4a, S7a–b and S18 require their original labels and captions when interpreted. Their presence does not make them current benchmark results.

![REV7 benchmark](figures/main/Figure4.png)

## Verify the package

Run from the repository root with Python 3.10 or newer:

```bash
python tools/verify_package.py
```

This checks file hashes, table and figure coverage, Python syntax and local documentation links. It does not train models or validate the scientific analyses.

To regenerate the CSV exports from the supplied workbook:

```bash
python -m pip install -r requirements-tools.txt
python tools/export_supplementary_tables.py
```

The export utility reads the workbook without changing it. If exports are regenerated or other files are changed, the saved checksum manifest must be refreshed before its integrity check can pass.

## Reproduction status

The August source snapshot and September reconstruction scripts use different inputs and protocols from the corrected five-seed REV7 benchmark. Their original dependency files are retained inside their respective legacy directories. A historical script with “leakfree” or “corrected” in its filename is not evidence that it produced the current results.

The corrected v2.0-review release should be completed with its matching code, inputs, folds, transformations, predictions, weights, environment and barcode-level audit before being described as a reproducible revision archive. No archive DOI or published v2.0-review URL has been assigned in this package.

## Citation and license

Repository: [jieli706/breast](https://github.com/jieli706/breast). Citation metadata are in [`CITATION.cff`](CITATION.cff). The snapshot identifier `REV7-materials-20261013` identifies these assembled materials; it is not a declaration that the corrected software release is complete.

The source packages did not contain a final software license. Their status is recorded in [`LICENSE_PENDING.txt`](LICENSE_PENDING.txt). Provider datasets retain their own access and reuse terms.
