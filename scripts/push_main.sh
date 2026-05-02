#!/usr/bin/env bash
set -euo pipefail

git branch -M main
git status --short
git push -u origin main
