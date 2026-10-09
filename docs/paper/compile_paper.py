#!/usr/bin/env python3
"""Compile this multi-file paper with Tectonic, including its bibliography."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    paper_dir = Path(__file__).resolve().parent
    bundled = Path("/Applications/ChatGPT.app/Contents/Resources/tectonic/tectonic")
    executable = shutil.which("tectonic") or (str(bundled) if bundled.is_file() else None)
    if executable is None:
        sys.exit("Tectonic is unavailable; the macOS app normally bundles it.")

    env = os.environ.copy()
    env["TECTONIC_UNTRUSTED_MODE"] = "1"
    app_cache = Path.home().resolve() / "Library/Application Support/ChatGPT/latex/tectonic"
    if app_cache.is_dir():
        env.setdefault("TECTONIC_CACHE_DIR", str(app_cache))

    return subprocess.run(
        [executable, "-X", "compile", "--untrusted", "--keep-logs",
         "--outdir", str(paper_dir), "jsc_paper_main.tex"],
        cwd=paper_dir, env=env, check=False,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
