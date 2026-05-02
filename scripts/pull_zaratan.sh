#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-zaratan}"
REMOTE_DIR="${2:-/home/gnanesh/scratch.msml612pcs3/latent_void}"

tmux send-keys -t "${TARGET}" "cd ${REMOTE_DIR} && git pull --ff-only && git status --short --branch" C-m
