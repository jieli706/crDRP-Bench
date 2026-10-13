# Reproduction status for the REV7 materials

Prepared 13 October 2026. The numerical tables and figure files match the locally available REV7 documents. Full computational reproduction of the corrected analyses remains incomplete.

## Available in this snapshot

- The original supplementary workbook with 63 worksheets and 62 CSV table exports.
- Six main figures and 18 supplementary figures, each in PDF and PNG format.
- Captions and supplementary-methods text from the clean REV7 documents.
- The August source repository and September local revision scripts, explicitly archived as historical code.
- File hashes, table-export code and a package-integrity checker.

The PNG figures match the images embedded in the clean REV7 manuscript and supplementary information byte for byte. The PDF counterparts were selected from the same supplied figure versions. REV7 changes to Figures 1, 3, 6, S3, S5 and S14 were local edits to vector text or layout; this assembly did not rerun the underlying model analyses.

## Files still needed for the corrected release

| Required material | Needed information | Status here |
| --- | --- | --- |
| Corrected revision pipeline | Complete scripts and run commands for compound correction, tensor alignment, leak-free fitting, controls, inference and figures | Missing |
| Corrected inputs | Ordered sample, gene and drug IDs; corrected SMILES and feature-library membership; processed tensors and labels, or executable construction steps | Missing |
| Saved data splits | Train/validation/test membership for every seed, fold and holdout regime, plus pretraining exclusions | Missing |
| Fitted transformations | Matching scalers, PCA bases, fitted subsets and feature ordering | Missing |
| Out-of-fold predictions | Stable pair IDs, seed, fold, regime, model/control arm, observed response and prediction; matching current bootstrap inputs | Missing |
| Model weights | Corrected pretrained and fine-tuned weights with architecture/configuration and checksums | Missing |
| Barcode-level atlas record | Per-barcode QC, cluster and compartment assignments supporting Table S15 | Missing; summary tables only |
| Run environment | Exact environment/package lock files, run configuration and relevant training logs for the corrected analyses | Missing; legacy environment files only |
| Versioned archive | Matching GitHub release URL and a deposited archive DOI | Not supplied |

These items were not found in the inspected local REV6/REV7 packages or older local code/evidence archives. This does not establish whether they are present on C2B2 or in another external location.

## What can be checked now

The supplied workbook can be compared with its CSV exports. File hashes and figure coverage can be checked locally. Reported contrasts can be read from Table S1e, compound-identity corrections from Tables S3e–S3f, and cell-count flow from Table S15.

Those summaries do not replace the raw predictions, fitted transformations or model weights. The package checker reports archive completeness separately from file integrity.

## Completing v2.0-review

Place the actual corrected revision scripts under `scripts/revision/` and document their dependencies and commands. Add the matching current numerical inputs or construction recipes, splits, transformations and prediction files. Store larger permitted inputs and weights in release/archive assets and record their locations and hashes.

Recompute the reported current summaries and figures from the matching predictions, then compare with the provided REV7 tables. Preserve the historical labels for superseded audit material. Replace the availability fields in `release_status.json` only after the missing files have been recovered and checked. Refresh `manifest.sha256` after changes.

Assign the final software release identifier and DOI only to the completed matching archive. The draft preparation identifier in this snapshot should not be cited as a completed v2.0-review computation.
