"""Record exact library versions at runtime into ENVIRONMENT.md (reproducibility requirement)."""
from __future__ import annotations

import platform
from importlib import import_module
from pathlib import Path

PACKAGES = ["numpy", "scipy", "sklearn", "torch", "pandas", "matplotlib",
            "umap", "pymde", "pcc", "openTSNE", "toorpia"]


def _version(name: str) -> str:
    try:
        mod = import_module(name)
        return getattr(mod, "__version__", "installed (version unknown)")
    except Exception as e:  # noqa: BLE001
        return f"NOT INSTALLED ({type(e).__name__})"


def collect_versions() -> dict:
    return {name: _version(name) for name in PACKAGES}


def write_environment(path) -> Path:
    path = Path(path)
    vers = collect_versions()
    lines = ["# Environment", "",
             f"- python: {platform.python_version()} ({platform.python_implementation()})",
             f"- platform: {platform.platform()}", ""]
    lines.append("| package | version |")
    lines.append("|---|---|")
    for name, v in vers.items():
        lines.append(f"| {name} | {v} |")
    lines += ["",
              "Recorded automatically by `run/environment.py` at benchmark runtime. "
              "Pinned versions are in `requirements.txt`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
