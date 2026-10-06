"""Provenance and the overwrite guard for every script in scripts/.

Each script's subcommand dispatcher calls `provenance.main(COMMANDS, __doc__)`. That
installs a Python audit hook which sees every file the process opens, and:

- **refuses to overwrite** an existing file under artifacts/ unless the command line
  carries `--overwrite` (stripped before the subcommand parses its arguments). A file
  this process has already written may be rewritten (scripts that save after every
  model). Checkpoint directories (any directory whose name contains "ckpt") are exempt:
  resuming and extending into them is their purpose.
- **records what was read**: the sha256 of every file under the repo opened for reading,
  except code (scripts/, utils/, tests/, which the git commit covers).
- **records what was written**: at exit, every file written under the repo gets a
  sidecar `<file>.prov`; files in a checkpoint or cache directory instead get one
  line per run in that directory's `PROVENANCE.jsonl`.

A record holds the command (repo-relative), the git commit and whether the code was
dirty, the inputs and their hashes, the output's own hash, the environment overrides
in effect, library versions, start and end times, and whether the command finished.
Paths are repo-relative; files outside the repo (model weights, the HF cache) are not
recorded, since the command names the model.

Drafted with the assistance of Claude (Anthropic).
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts"
CODE_DIRS = ("scripts", "utils", "tests", ".git")
SIDECAR = ".prov"          # not ".json": a "*.json" glob over artifacts must never pick up a record
DIR_LOG = "PROVENANCE.jsonl"
# environment variables some scripts or truthlib read as overrides; recorded when set
ENV_OVERRIDES = ("STEER_CKPT", "FIGURE_DIR", "SNR_SWEEP", "ROGUE_JSON", "CHI_AGG", "EMERGENCE_DS",
                 "TRANSFER_MODEL", "TRANSFER_FULL", "HF_MODELS_ROOT", "EXTERNAL_ROOT",
                 "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "PYTORCH_ENABLE_MPS_FALLBACK")

_state: dict = {"installed": False}


class OverwriteRefused(PermissionError):
    pass


def _rel(p: Path) -> str | None:
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return None


def _is_code(rel: str) -> bool:
    return rel.split("/", 1)[0] in CODE_DIRS or rel.endswith((".py", ".pyc"))


def _in_ckpt_dir(p: Path) -> bool:
    return any("ckpt" in part for part in p.relative_to(REPO).parts[:-1])


def _aggregated(p: Path) -> bool:
    """Outputs logged per directory rather than with a sidecar each."""
    parts = p.relative_to(REPO).parts[:-1]
    return any("ckpt" in part or part == "act_cache" for part in parts)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:                       # the hook ignores reads while hashing
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _writes(mode, flags) -> bool:
    if isinstance(mode, str):
        return any(c in mode for c in "wax+")
    if isinstance(flags, int):
        return bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    return False


def _hook(event: str, args: tuple) -> None:
    if event != "open" or _state.get("busy"):
        return
    path, mode, flags = (tuple(args) + (None, None, None))[:3]
    if not isinstance(path, (str, bytes, os.PathLike)):
        return                                      # an fd
    p = Path(os.fsdecode(path)).absolute()
    rel = _rel(p)
    if rel is None or _is_code(rel) or p.name.endswith(SIDECAR) or p.name == DIR_LOG:
        return
    _state["busy"] = True
    try:
        if _writes(mode, flags):
            if p not in _state["written"]:
                if (p.exists() and not _state["overwrite"] and rel.startswith("artifacts/")
                        and not _in_ckpt_dir(p)):
                    raise OverwriteRefused(f"{rel} exists; rerun with --overwrite to replace it")
                _state["written"].append(p)
        elif p not in _state["read"] and p not in _state["written"] and p.is_file():
            _state["read"][p] = sha256(p)
    finally:
        _state["busy"] = False


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def _portable(a: str) -> str:
    """A repo-relative path, or <external>/name for an absolute path outside the repo:
    no machine paths in a record."""
    if os.path.isabs(a):
        r = _rel(Path(a))
        return r if r is not None else os.path.join("<external>", os.path.basename(a))
    return a


def _command() -> list[str]:
    return [_portable(a) for a in _state["argv"]]


def _versions() -> dict:
    v = {"python": sys.version.split()[0]}
    for m in ("numpy", "scipy", "torch", "transformers", "sklearn", "matplotlib"):
        mod = sys.modules.get(m)
        if mod is not None:
            v[m] = getattr(mod, "__version__", "?")
    return v


def _record(status: str) -> None:
    if not _state["written"]:
        return
    _state["busy"] = True
    # not _git(): its strip() would eat the leading space of the first porcelain line (" M path")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "scripts", "utils"], cwd=REPO,
                           capture_output=True, text=True).stdout.rstrip("\n")
    base = {
        "command": ["python"] + _command(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty),
        "dirty_files": [ln[3:] for ln in dirty.splitlines()],
        "started": _state["started"], "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status,
        "env": {k: _portable(os.environ[k]) for k in ENV_OVERRIDES if k in os.environ},
        "versions": _versions(),
        "inputs": {_rel(p): h for p, h in sorted(_state["read"].items(), key=lambda kv: str(kv[0]))},
    }
    by_dir: dict[Path, list] = {}
    for p in _state["written"]:
        if not p.is_file():
            continue
        if _aggregated(p):
            by_dir.setdefault(p.parent, []).append(p)
            continue
        rec = {"artifact": _rel(p), "sha256": sha256(p), **base}
        Path(str(p) + SIDECAR).write_text(json.dumps(rec, indent=1) + "\n")
    for d, files in by_dir.items():
        rec = {**base, "outputs": {p.name: sha256(p) for p in sorted(files)}}
        with open(d / DIR_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
    _state["busy"] = False


def install(argv: list[str]) -> list[str]:
    """Install the hook; returns argv with --overwrite removed."""
    if _state["installed"]:
        raise RuntimeError("provenance.install called twice")
    overwrite = "--overwrite" in argv
    argv = [a for a in argv if a != "--overwrite"]
    _state.update(installed=True, overwrite=overwrite, read={}, written=[], busy=False,
                  argv=[os.path.relpath(argv[0], REPO) if os.path.isabs(argv[0]) else argv[0]] + argv[1:],
                  started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    sys.addaudithook(_hook)
    return argv


def main(commands: dict, doc: str) -> None:
    """The dispatcher every script uses: `python scripts/x.py <subcommand> [args] [--overwrite]`."""
    argv = install(list(sys.argv))
    if len(argv) < 2 or argv[1] not in commands:
        print(doc)
        sys.exit(0 if argv[1:2] in (["-h"], ["--help"]) else 2)
    cmd = argv[1]
    sys.argv = [f"{Path(argv[0]).name} {cmd}"] + argv[2:]   # argparse usage names the subcommand
    status = "failed"
    atexit.register(lambda: _record(_state.get("status", status)))
    try:
        commands[cmd](argv[2:])
        _state["status"] = "ok"
    except SystemExit as e:
        _state["status"] = "ok" if e.code in (0, None) else f"exit {e.code}"
        raise
    except BaseException as e:
        _state["status"] = f"failed: {type(e).__name__}"
        raise


def run(fn) -> None:
    """Entry point for a single-command script: `provenance.run(main)` under `__main__`.
    `fn` parses sys.argv itself; --overwrite is removed from it first."""
    sys.argv = install(list(sys.argv))
    status = "failed"
    atexit.register(lambda: _record(_state.get("status", status)))
    try:
        fn()
        _state["status"] = "ok"
    except SystemExit as e:
        _state["status"] = "ok" if e.code in (0, None) else f"exit {e.code}"
        raise
    except BaseException as e:
        _state["status"] = f"failed: {type(e).__name__}"
        raise
