# REV7 figure text and layout edits

`edit_figures_original.py` is the unmodified local script used to update the vector text or layout of Figures 1, 3, 6, S3, S5 and S14 in the supplied REV7 package. `figure_edit_log.json` preserves the recorded text changes, geometry changes and input/output PDF hashes. Workstation-specific source paths are replaced with figure identifiers in the log.

This script edits existing PDF content and rasterizes the resulting figures. It does not train models or regenerate the scientific analyses. Its original configuration file, input-directory layout and Windows Poppler location must be supplied or adapted before execution; it is archived here as source provenance.

The current assembled figure files are under [`../../figures/`](../../figures/). The matching corrected training scripts and predictions remain unavailable, as documented in [`../../REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md).
