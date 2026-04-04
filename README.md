# icemodels

Example/demo in https://github.com/keflavich/icemodels/blob/main/notebooks/CO2_Phoenix_Example.ipynb


Docs at https://keflavich.github.io/icemodels/

## Notebook CI and output stripping

Notebook execution timing is checked in GitHub Actions by
`.github/workflows/docs.yml` using `docs/check_notebook_execution_times.py`.
Cells that take at least 30 seconds are flagged as `slow` in
`docs/notebook_timing_report.json` and in the workflow summary.

To keep local notebooks live while committing stripped notebooks:

1. Install hooks:

	```bash
	pre-commit install
	nbstripout --install --attributes .gitattributes
	```

2. Keep working normally in notebooks. The `*.ipynb filter=nbstripout` rule in
	`.gitattributes` strips output for the git index during add/commit, while
	your local working copy keeps outputs and images.


![Ices Measure Metals](metalice.png)
