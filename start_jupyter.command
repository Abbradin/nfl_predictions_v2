#!/bin/bash
# Run from any working directory: double-click or ./start_jupyter.command
set -e
cd "$(dirname "$0")"
printf "\n=== NFL Prediction Model V2 | Local Jupyter ===\n"
if [ ! -d .venv ]; then
  if command -v python3.13 >/dev/null 2>&1; then PYTHON=python3.13
  elif command -v python3 >/dev/null 2>&1; then PYTHON=python3
  else echo "Python 3 is required. Install Python 3.13 and retry."; read -r -p "Press Enter to close..."; exit 1
  fi
  echo "Creating .venv using $PYTHON..."
  "$PYTHON" -m venv .venv
fi
source .venv/bin/activate
# Install only on initial setup; use requirements file as installation marker.
if ! python -c 'import jupyterlab, streamlit, lightgbm, nflreadpy, pyarrow' >/dev/null 2>&1; then
  echo "Installing project requirements and JupyterLab (first run can take a few minutes)..."
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt 'jupyterlab>=4,<5' 'ipykernel>=6,<8'
fi
python check_artifacts.py
printf "\nOpening JupyterLab in your browser. Open NFL_V2_Local_Workbench.ipynb.\n"
printf "Stop server with Control-C in this Terminal window.\n\n"
exec python -m jupyterlab --notebook-dir="$(pwd)" --ServerApp.open_browser=True
