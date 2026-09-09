"""
mzML -> QC feature table.

Streams the file, keeps only the most recent MS1 scan in memory, and writes
Parquet in chunks so a two-hour gradient does not sit in RAM.
"""

from __future__ import annotations

import os
import re
from typing import Iterator

import numpy as np
import pandas as pd
from pyteomics import mzml

from .features import spectrum_features

SCAN_RE = re.compile(r"scan=(\d+)")


def _scan_number(spectrum_id: str, fallback: int) -> int:
    m = SCAN_RE.search(spectrum_id or "")
    return int(m.group(1)) if m else fallback


def _precursor_info(spec):
    """Pull precursor m/z, charge, isolation window from an mzML MS2 entry."""
    try:
        prec = spec["precursorList"]["precursor"][0]
    except (KeyError, IndexError):
        return None
    try:
        ion = prec["selectedIonList"]["selectedIon"][0]
    except (KeyError, IndexError):
        return None

    mz = float(ion.get("selected ion m/z", 0.0))
    charge = int(ion.get("charge state", 0) or 0)
    prec_int = float(ion.get("peak intensity", 0.0) or 0.0)

    iso = prec.get("isolationWindow", {})
    target = float(iso.get("isolation window target m/z", mz) or mz)
    lower = float(iso.get("isolation window lower offset", 0.7) or 0.7)
    upper = float(iso.get("isolation window upper offset", 0.7) or 0.7)

    return {
        "precursor_mz": mz or target,
        "charge": charge,
        "precursor_intensity": prec_int,
        "iso_target": target,
        "iso_lower": lower,
        "iso_upper": upper,
    }


def _scan_meta(spec):
    try:
        scan = spec["scanList"]["scan"][0]
    except (KeyError, IndexError):
        return {}
    rt = scan.get("scan start time", np.nan)
    return {
        "rt_min": float(rt) if rt is not None else np.nan,
        "injection_time_ms": float(scan.get("ion injection time", np.nan) or np.nan),
    }


def iter_ms2_features(path: str, run_id: str | None = None,
                      max_peaks: int = 150, frag_tol: float = 0.02,
                      keep_peaks: bool = False,
                      peak_cap: int = 200) -> Iterator[dict]:
    """
    Yield one feature dict per MS2 spectrum.

    keep_peaks=True also carries the top peaks through, which the HTML report
    needs. Turn it off for large-scale batch runs.
    """
    run_id = run_id or os.path.splitext(os.path.basename(path))[0]
    last_ms1 = None
    n_missing_charge = 0

    with mzml.read(path, use_index=False) as reader:
        for i, spec in enumerate(reader):
            level = spec.get("ms level")
            mz = spec.get("m/z array")
            inten = spec.get("intensity array")
            if mz is None or inten is None or mz.size == 0:
                continue

            if level == 1:
                last_ms1 = (np.asarray(mz), np.asarray(inten))
                continue
            if level != 2:
                continue

            pinfo = _precursor_info(spec)
            if pinfo is None:
                continue

            charge = pinfo["charge"]
            charge_imputed = False
            if charge == 0:
                charge = 2
                charge_imputed = True
                n_missing_charge += 1

            feats = spectrum_features(
                np.asarray(mz), np.asarray(inten),
                pinfo["precursor_mz"], charge,
                ms1=last_ms1,
                iso_lower=pinfo["iso_lower"], iso_upper=pinfo["iso_upper"],
                max_peaks=max_peaks, frag_tol=frag_tol,
            )
            if feats is None:
                continue

            feats["run_id"] = run_id
            feats["scan_number"] = _scan_number(spec.get("id", ""), i)
            feats["spectrum_id"] = spec.get("id", f"index={i}")
            feats["charge_imputed"] = charge_imputed
            feats["precursor_intensity"] = pinfo["precursor_intensity"]
            feats.update(_scan_meta(spec))

            if keep_peaks:
                from .features import top_n
                kmz, kint = top_n(np.asarray(mz), np.asarray(inten), peak_cap)
                feats["_mz"] = np.round(kmz, 4).tolist()
                feats["_intensity"] = np.round(
                    kint / max(kint.max(), 1e-9) * 100, 2).tolist()

            yield feats

    if n_missing_charge:
        print(f"  [warn] {run_id}: {n_missing_charge} MS2 scans had no charge "
              f"state in the mzML; assumed 2+ and flagged as charge_imputed")


def extract_run(path: str, run_id: str | None = None, keep_peaks: bool = False,
                **kwargs) -> pd.DataFrame:
    rows = list(iter_ms2_features(path, run_id, keep_peaks=keep_peaks, **kwargs))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    front = ["run_id", "scan_number", "spectrum_id", "rt_min",
             "precursor_mz", "charge"]
    cols = front + [c for c in df.columns if c not in front]
    return df[cols]


def inspect_mzml(path: str, n_probe: int = 400) -> dict:
    """
    Cheap structural inspection of an mzML before committing to a full run.

    Answers the questions that decide whether the QC features will mean
    anything: is it centroided, are charge states present, are isolation
    windows recorded, and what is the run_id going to be.
    """
    from pyteomics import mzml as pymzml

    run_id = os.path.splitext(os.path.basename(path))[0]
    info = {"path": path, "run_id": run_id, "n_ms1": 0, "n_ms2": 0,
            "centroided": None, "profile": 0, "scan_min": None,
            "scan_max": None, "n_charge_missing": 0, "n_no_isolation": 0,
            "median_peaks": None, "rt_min": None, "rt_max": None,
            "id_style": None}
    peak_counts, scans, rts = [], [], []

    info["truncated"] = not check_truncation(path)["complete"]

    reader = pymzml.read(path, use_index=False)
    i = -1
    while True:
        # A torn file must not stop the probe from reporting what it did see.
        try:
            spec = next(reader)
        except StopIteration:
            break
        except Exception:
            info["truncated"] = True
            break
        i += 1
        if True:
            if i >= n_probe:
                break
            lvl = spec.get("ms level")
            if info["id_style"] is None:
                info["id_style"] = str(spec.get("id", ""))[:80]
            if "profile spectrum" in spec:
                info["profile"] += 1
            if lvl == 1:
                info["n_ms1"] += 1
                continue
            if lvl != 2:
                continue
            info["n_ms2"] += 1
            peak_counts.append(int(np.asarray(spec.get("m/z array", [])).size))
            sn = _scan_number(spec.get("id", ""), i)
            if sn is not None:
                scans.append(sn)
            try:
                rts.append(float(spec["scanList"]["scan"][0]["scan start time"]))
            except Exception:
                pass
            try:
                prec = spec["precursorList"]["precursor"][0]
                sel = prec["selectedIonList"]["selectedIon"][0]
                if not sel.get("charge state"):
                    info["n_charge_missing"] += 1
                if "isolationWindow" not in prec:
                    info["n_no_isolation"] += 1
            except Exception:
                info["n_charge_missing"] += 1

    if peak_counts:
        info["median_peaks"] = float(np.median(peak_counts))
        info["centroided"] = info["profile"] == 0
    if scans:
        info["scan_min"], info["scan_max"] = int(min(scans)), int(max(scans))
    if rts:
        info["rt_min"], info["rt_max"] = float(min(rts)), float(max(rts))
    return info


def check_truncation(path: str, tail_bytes: int = 8192) -> dict:
    """
    Is this mzML complete?

    A well-formed mzML ends with </mzML>, and an indexed one with
    </indexedmzML>. Truncated files are common: an interrupted conversion, a
    killed download, a browser upload that stopped short, or a job that hit a
    disk quota. They parse fine for hundreds of thousands of lines and then
    throw XMLSyntaxError deep inside the run, which looks like a bug in the
    reader rather than a bad input.

    Checking the tail costs one seek, so there is no reason not to.
    """
    out = {"complete": False, "has_index": False, "size_mb": 0.0,
           "detail": ""}
    try:
        size = os.path.getsize(path)
        out["size_mb"] = size / 1e6
        with open(path, "rb") as fh:
            fh.seek(max(0, size - tail_bytes))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError as e:
        out["detail"] = f"could not read the file: {e}"
        return out

    out["has_index"] = "</indexedmzML>" in tail
    if "</mzML>" in tail or out["has_index"]:
        out["complete"] = True
        return out

    stub = tail.strip()[-70:].replace("\n", " ")
    out["detail"] = (f"file ends without a closing </mzML> tag. Last bytes: "
                     f"...{stub}")
    return out


def iter_ms2_features_safe(path, run_id=None, on_error=None, **kwargs):
    """
    Wrap iter_ms2_features so a malformed tail does not discard the spectra
    that were read successfully.

    Salvaging is the right default here: a file truncated at 90% still holds
    90% of a usable run, and losing it to an exception helps nobody. The
    caller is told exactly how many spectra were recovered so the result is
    never mistaken for a complete run.
    """
    from lxml.etree import XMLSyntaxError

    n = 0
    it = iter_ms2_features(path, run_id, **kwargs)
    while True:
        try:
            row = next(it)
        except StopIteration:
            break
        except XMLSyntaxError as e:
            if on_error:
                on_error(n, str(e).split("(")[0].strip())
            break
        except Exception as e:  # a corrupt binary array, not a torn file
            if on_error:
                on_error(n, f"{type(e).__name__}: {e}")
            break
        n += 1
        yield row


def extract_run_safe(path: str, run_id: str | None = None, **kwargs):
    """
    Returns (dataframe, warning_or_None). Never raises on a truncated file.
    """
    problem = {}

    def note(n, msg):
        problem["n"] = n
        problem["msg"] = msg

    rows = list(iter_ms2_features_safe(path, run_id, on_error=note, **kwargs))
    df = pd.DataFrame(rows)
    if not problem:
        return df, None
    warn = (f"{os.path.basename(path)} is truncated or corrupt. Recovered "
            f"{problem['n']:,} MS2 spectra before parsing failed "
            f"({problem['msg']}). Treat this run as INCOMPLETE: "
            f"identification rate and every per-run fraction are computed "
            f"only over the part that was readable, and the missing tail is "
            f"usually the end of the gradient.")
    return df, warn
