#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/gnanesh/scratch.msml612pcs3/latent_void}"
VENV="${VENV:-${ROOT}/.venvs/latent_void_py310}"
ZIP="${ROOT}/data/downloads/inpaint360.zip"

cd "${ROOT}"
unset PYTHONPATH
source "${VENV}/bin/activate"
mkdir -p data/downloads data

if [[ ! -s "${ZIP}" ]]; then
  gdown 1YLpop12JRbzglJfx0FUFUZ2GLaBfZX_x -O "${ZIP}" --continue
else
  echo "[exists] ${ZIP}"
fi

if [[ ! -d data/inpaint360 ]]; then
  UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q -n "${ZIP}" -d data
else
  echo "[exists] data/inpaint360"
fi

find data/inpaint360 -maxdepth 1 -mindepth 1 -type d | sort | sed -n '1,80p'
