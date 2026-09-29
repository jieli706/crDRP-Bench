# crDRP-Bench: Quantifying Compartment Signal Recovery in a Breast-Reference Drug-Response Benchmark

This repository accompanies the revised manuscript. It separates the availability of compartment-dependent signal, recovery of that signal from estimated inputs, and prediction of measured breast cancer cell-line drug responses.

## Repository contents

- `code/pipeline/`: analysis and data-preparation scripts from the study pipeline.
- `code/revision_analyses/`: scripts for the added donor-mixture, NNLS, BayesPrism-updated, TCGA, carcinoma-source, PCA and XGB analyses, plus figure generation.
- `figures/`: the five main figures and eight supplementary figures in PNG and PDF formats.
- `Data_Sheet_1_260929.zip`: numerical tables, per-sample/drug records, splits and supporting analysis files.

The manuscript, response letter, tracked/blue revision copy and reviewer-only historical crosswalk are not included in this code repository. The full reproducibility archive is kept separately because it is large and contains intermediate project folders; review and clean it before publishing as a release asset or depositing it in a research-data archive.

## Data and reproducibility

The analyses use public DepMap, GDSC, PRISM, ChEMBL, CELLxGENE/GSE176078 and TCGA resources. The data sheet records release/accession information, source URLs and file hashes. Obtain source data from the original providers and follow their current access and redistribution terms. Derived outputs in the data sheet do not replace the source data.

The scripts expect the study's relative project hierarchy and the matching Python/R packages. The full inputs, transformations, model checkpoints, predictions and environment records are in the separate reproducibility archive; they are not stored in this GitHub-ready folder. See that archive's README and SHA-256 manifest when preparing the final archived deposit. The main summary estimates and limitations are described in the accompanying manuscript.

The mixture and positive-control response labels are synthetic. PRISM validation evaluates frozen GDSC model predictions on overlapping cell/drug keys; it is not an independent patient cohort. TCGA purity and survival analyses are observational.

## Before making the repository public

1. Add an author-approved license; `LICENSE_PENDING.txt` is a reminder, not a license.
2. Check that the authors may redistribute each included derived dataset and figure.
3. Add the final DOI and versioned archive link to this README after the archive has actually been published. No DOI is assigned in this bundle.
4. Inspect `code/pipeline/` for scripts that require local project paths and set up a clean environment before documenting an end-to-end command.

## Citation

Cheng X, Yang J, Li H, Sun Y, Li J. *crDRP-Bench: Quantifying Compartment Signal Recovery in a Breast-Reference Drug-Response Benchmark*. Version 2026-09-29. Add the final persistent archive DOI here after publication.
