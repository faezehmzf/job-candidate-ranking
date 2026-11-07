#!/usr/bin/env bash
# Reproduce every file in results/. Run from the repository root (about 15 minutes on a laptop).
set -euo pipefail
PY=${PY:-python}
$PY scripts/01_data_audit.py        # data-quality audit + dataset facts
$PY scripts/02_compare_models.py    # 8 models, 5-fold GroupKFold on job_id
$PY scripts/03_ablation.py          # feature-group ablation
$PY scripts/04_select_and_test.py   # model selection by CV, then one evaluation on the test set
$PY -m pytest -q tests
