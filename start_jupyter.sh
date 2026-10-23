#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install -r requirements.txt 'jupyterlab>=4,<5' 'ipykernel>=6,<8'
python check_artifacts.py
exec python -m jupyterlab --notebook-dir="$(pwd)"
