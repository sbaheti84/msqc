#!/usr/bin/env bash
# Launch the msqc dashboard.
#
#   ./run_app.sh                              # pick the file in the sidebar
#   ./run_app.sh results/qc_triaged.parquet   # open it directly
#
# Needs: pip install -e ".[app]"
set -euo pipefail

QC="${1:-}"
if [ -n "$QC" ]; then
  exec streamlit run app.py -- --qc "$QC"
else
  exec streamlit run app.py
fi
