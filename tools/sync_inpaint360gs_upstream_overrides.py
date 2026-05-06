#!/usr/bin/env python3
"""Copy patched Inpaint360GS files into upstream_overrides/ for git commits.

external/Inpaint360GS/ is gitignored; we keep verbatim copies here so changes are not lost.
Run this on any machine where external/Inpaint360GS is populated, then git add upstream_overrides/.

Usage:
    python tools/sync_inpaint360gs_upstream_overrides.py
    git add upstream_overrides/inpaint360gs/
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    src_extra = Path(os.environ.get("INPAINT360GS_COPY_SRC", root / "external" / "Inpaint360GS")).resolve()
    src_d = src_extra if src_extra.is_dir() else (root / "external" / "Inpaint360GS").resolve()

    dst_d = root / "upstream_overrides" / "inpaint360gs"
    dst_d.mkdir(parents=True, exist_ok=True)

    files = ("edit_object_removal_plyfusion.py", "edit_object_inpaint.py")
    for name in files:
        src_f = src_d / name
        if not src_f.is_file():
            print(f"ERROR: missing {src_f}\nPopulate external/Inpaint360GS or set INPAINT360GS_COPY_SRC=", flush=True)
            return 1
        dst_f = dst_d / name
        shutil.copy2(src_f, dst_f)
        print(f"OK {dst_f.relative_to(root)} ({dst_f.stat().st_size} bytes)", flush=True)
    print("Done. Next: git add upstream_overrides/inpaint360gs/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
