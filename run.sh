#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m bookrec.pipeline
.venv/bin/python -m pytest -q
.venv/bin/python scripts/build_report.py
if [[ "${1:-}" == "--serve" ]]; then
  .venv/bin/python -m uvicorn bookrec.api:app --host 127.0.0.1 --port 8000
fi
