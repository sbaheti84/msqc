"""
Thermo RAW to mzML conversion.

No Python library reads Thermo RAW directly, so this shells out to whichever
converter is installed. Backends are tried in order of reliability rather than
speed, because a silently wrong conversion is far more expensive than a slow
one.

Conversion settings that matter, and why they are not configurable:

* **Centroided output.** Profile-mode data makes every peak count, noise
  estimate and signal-to-noise feature in this package meaningless. Both
  backends peak-pick by default and this module keeps that default.
* **Indexed mzML.** Costs nothing and lets downstream tools seek.
* **No zlib on the ThermoRawFileParser path.** Compression saves disk but
  costs parse time on every subsequent run, and these files get read
  repeatedly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable

RAW_SUFFIXES = (".raw",)
MZML_SUFFIXES = (".mzml",)

# Pinned rather than :latest. A converter that changes under you is a
# reproducibility problem, and the diff shows up as unexplained metric drift.
TRFP_IMAGE = ("quay.io/biocontainers/thermorawfileparser:"
              "1.4.5--ha8f3691_0")
PWIZ_IMAGE = "chambm/pwiz-skyline-i-agree-to-the-vendor-licenses"


def is_raw(path: str) -> bool:
    return path.lower().endswith(RAW_SUFFIXES)


def is_mzml(path: str) -> bool:
    return path.lower().endswith(MZML_SUFFIXES)


def _docker_ok() -> bool:
    exe = shutil.which("docker")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "info"], capture_output=True,
                              timeout=20).returncode == 0
    except Exception:
        return False


def available_backends() -> list[dict]:
    """
    What can this machine actually do, right now. Each entry has an id, a
    human label, and whether it is usable.
    """
    out = []

    native = (shutil.which("thermorawfileparser")
              or shutil.which("ThermoRawFileParser.sh")
              or shutil.which("ThermoRawFileParser"))
    out.append({
        "id": "trfp_native", "ok": bool(native), "exe": native,
        "label": "ThermoRawFileParser (installed locally)",
        "install": "conda install -c bioconda thermorawfileparser"})

    mono = shutil.which("mono")
    dll = os.environ.get("THERMORAWFILEPARSER_DLL", "")
    out.append({
        "id": "trfp_mono", "ok": bool(mono and dll and os.path.exists(dll)),
        "exe": mono, "dll": dll,
        "label": "ThermoRawFileParser via mono",
        "install": ("brew install mono, download ThermoRawFileParser.exe from "
                    "github.com/compomics/ThermoRawFileParser/releases, then "
                    "export THERMORAWFILEPARSER_DLL=/path/to/"
                    "ThermoRawFileParser.exe")})

    docker = _docker_ok()
    out.append({
        "id": "trfp_docker", "ok": docker,
        "label": "ThermoRawFileParser in Docker (recommended)",
        "install": "install Docker Desktop and start it"})
    out.append({
        "id": "msconvert_docker", "ok": docker,
        "label": "ProteoWizard msconvert in Docker",
        "install": "install Docker Desktop and start it"})

    msconvert = shutil.which("msconvert")
    out.append({
        "id": "msconvert_native", "ok": bool(msconvert), "exe": msconvert,
        "label": "msconvert (installed locally)",
        "install": "install ProteoWizard and put msconvert on PATH"})
    return out


def best_backend() -> str | None:
    order = ["trfp_native", "trfp_docker", "trfp_mono",
             "msconvert_native", "msconvert_docker"]
    ok = {b["id"]: b for b in available_backends() if b["ok"]}
    for b in order:
        if b in ok:
            return b
    return None


def _build_command(backend, raw_path, out_dir):
    raw_dir = os.path.dirname(os.path.abspath(raw_path)) or "."
    raw_name = os.path.basename(raw_path)
    backends = {b["id"]: b for b in available_backends()}

    if backend == "trfp_native":
        # -f 2 is indexed mzML. Peak picking is on by default; do not add
        # --noPeakPicking, profile data breaks the QC features.
        return [backends[backend]["exe"], "-i", raw_path,
                "-b", os.path.join(out_dir, _mzml_name(raw_path)), "-f", "2"]

    if backend == "trfp_mono":
        return [backends[backend]["exe"], backends[backend]["dll"],
                "-i", raw_path,
                "-b", os.path.join(out_dir, _mzml_name(raw_path)), "-f", "2"]

    if backend == "trfp_docker":
        return ["docker", "run", "--rm",
                "-v", f"{raw_dir}:/in:ro", "-v", f"{out_dir}:/out",
                TRFP_IMAGE, "ThermoRawFileParser.sh",
                "-i", f"/in/{raw_name}",
                "-b", f"/out/{_mzml_name(raw_path)}", "-f", "2"]

    if backend == "msconvert_native":
        return [backends[backend]["exe"], raw_path, "--mzML",
                "--filter", "peakPicking vendor msLevel=1-",
                "-o", out_dir]

    if backend == "msconvert_docker":
        return ["docker", "run", "--rm",
                "-v", f"{raw_dir}:/in:ro", "-v", f"{out_dir}:/out",
                PWIZ_IMAGE, "wine", "msconvert", f"/in/{raw_name}",
                "--mzML", "--filter", "peakPicking vendor msLevel=1-",
                "-o", "/out"]

    raise ValueError(f"unknown backend {backend!r}")


def _mzml_name(raw_path: str) -> str:
    return os.path.splitext(os.path.basename(raw_path))[0] + ".mzML"


def convert_raw(raw_path: str, out_dir: str, backend: str | None = None,
                timeout: int = 7200, reuse: bool = True) -> str:
    """
    Convert one RAW to indexed, centroided mzML. Returns the mzML path.

    Reuses an existing conversion when the mzML is newer than the RAW, which
    matters because these files are large and conversion is the slowest step
    in the whole pipeline.
    """
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, _mzml_name(raw_path))

    if reuse and os.path.exists(dest):
        if os.path.getmtime(dest) >= os.path.getmtime(raw_path) \
                and os.path.getsize(dest) > 0:
            return dest

    backend = backend or best_backend()
    if backend is None:
        raise RuntimeError(
            "No RAW converter found on this machine. Install one of:\n"
            "  conda install -c bioconda thermorawfileparser   (simplest)\n"
            "  Docker Desktop, then msqc uses the pinned container\n"
            "Or convert the files yourself and load the mzML instead.")

    cmd = _build_command(backend, os.path.abspath(raw_path),
                         os.path.abspath(out_dir))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not os.path.exists(dest):
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        raise RuntimeError(
            f"Conversion of {os.path.basename(raw_path)} failed using "
            f"{backend}.\nCommand: {' '.join(cmd)}\n" + "\n".join(tail))
    return dest


def ensure_mzml(paths, work_dir: str, backend: str | None = None,
                progress: Callable[[float, str], None] | None = None):
    """
    Take a mixed list of RAW and mzML paths and return all-mzML.

    Returns (mzml_paths, conversions) where conversions maps the produced
    mzML back to its source RAW, so the UI can be honest about which files
    were converted rather than loaded as-is.
    """
    paths = list(paths)
    raws = [p for p in paths if is_raw(p)]
    out, conversions = [], {}

    for k, p in enumerate(paths):
        if not is_raw(p):
            out.append(p)
            continue
        if progress and raws:
            progress(k / max(len(paths), 1),
                     f"Converting {os.path.basename(p)} to mzML "
                     f"({raws.index(p) + 1} of {len(raws)})…")
        mz = convert_raw(p, work_dir, backend=backend)
        out.append(mz)
        conversions[mz] = p
    return out, conversions
