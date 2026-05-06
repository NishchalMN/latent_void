#!/usr/bin/env bash
# Copy tracked Inpaint360GS patches into external/Inpaint360GS (gitignored checkout).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/upstream_overrides/inpaint360gs"
DST="${ROOT}/external/Inpaint360GS"
if [[ ! -d "${DST}" ]]; then
  echo "ERROR: ${DST} not found. Clone or sync Inpaint360GS there first." >&2
  exit 1
fi
for f in edit_object_removal_plyfusion.py edit_object_inpaint.py; do
  if [[ ! -f "${SRC}/${f}" ]]; then
    echo "ERROR: missing tracked file ${SRC}/${f}; run python3 tools/sync_inpaint360gs_upstream_overrides.py" >&2
    exit 1
  fi
  cp -v "${SRC}/${f}" "${DST}/${f}"
done
echo "Installed upstream overrides into ${DST}"
