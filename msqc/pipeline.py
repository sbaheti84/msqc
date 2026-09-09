"""
The QC pipeline as a resumable, progress-reporting function.

The CLI and the Streamlit app both call `run_pipeline`. Keeping one
implementation means the app cannot drift from the command line, which is the
usual way a dashboard starts quietly reporting different numbers from the
batch job.
"""

from __future__ import annotations

import os
from typing import Callable, Iterable

import pandas as pd

from . import convert, extract, psm as psm_mod, rescue, score


def preflight(mzml_paths: Iterable[str], psm_paths: Iterable[str]):
    """
    Structural checks before any real work. Returns (info_list, problems).

    The join check is the one that matters. FragPipe writes the Spectrum
    column as <run>.<scan>.<scan>.<charge> and msqc derives run_id from the
    mzML filename, so a renamed or re-converted mzML silently joins to
    nothing: every spectrum reads as unassigned and the rescue pile fills
    with your best PSMs.
    """
    infos, problems = [], []
    raws = [p for p in mzml_paths if convert.is_raw(p)]
    if raws:
        b = convert.best_backend()
        if b is None:
            problems.append(
                f"{len(raws)} Thermo RAW file(s) given but no converter is "
                f"available on this machine. Install one with "
                f"`conda install -c bioconda thermorawfileparser`, or start "
                f"Docker Desktop, or convert to mzML yourself.")
        else:
            problems.append(
                f"NOTE: {len(raws)} RAW file(s) will be converted to "
                f"centroided indexed mzML using {b}. This is the slowest "
                f"step in the pipeline; results are cached and reused.")
    for p in [q for q in mzml_paths if not convert.is_raw(q)]:
        trunc = extract.check_truncation(p)
        if not trunc["complete"]:
            problems.append(
                f"{os.path.basename(p)} ({trunc['size_mb']:.0f} MB) is "
                f"TRUNCATED: {trunc['detail']} "
                f"The pipeline will salvage whatever parses, but the run is "
                f"incomplete. Re-convert or re-transfer the file. If it was "
                f"uploaded through the browser, upload usually is the cause "
                f"- use the 'Paths on this machine' option instead.")
        try:
            i = extract.inspect_mzml(p)
        except Exception as e:
            if trunc["complete"]:
                problems.append(
                    f"{os.path.basename(p)}: could not be read ({e}). "
                    f"Is it really mzML, and not mzXML or RAW?")
            continue
        infos.append(i)
        if i["n_ms2"] == 0:
            problems.append(f"{i['run_id']}: no MS2 scans found.")
        if i["centroided"] is False:
            problems.append(
                f"{i['run_id']}: profile-mode data. Peak counts and noise "
                f"estimates will be meaningless. Re-convert with peak picking "
                f"enabled.")
        if i["median_peaks"] and i["median_peaks"] > 2000:
            problems.append(
                f"{i['run_id']}: {i['median_peaks']:.0f} peaks per MS2 on "
                f"average, which suggests profile or unfiltered data.")
        if i["n_ms2"] and i["n_charge_missing"] / i["n_ms2"] > 0.15:
            problems.append(
                f"{i['run_id']}: {i['n_charge_missing'] / i['n_ms2']:.0%} of "
                f"MS2 scans carry no charge state. Mass-dependent features "
                f"fall back to z=2 for those.")
        if i["n_ms2"] and i["n_no_isolation"] / i["n_ms2"] > 0.1:
            problems.append(
                f"{i['run_id']}: isolation windows are missing, so "
                f"isolation_purity will be empty. That is the single most "
                f"useful feature here.")

    if psm_paths:
        try:
            allpsm = pd.concat([psm_mod.read_psm(p) for p in psm_paths],
                               ignore_index=True)
        except Exception as e:
            problems.append(f"PSM table could not be read: {e}")
            return infos, problems
        psm_runs = set(allpsm["run_id"].dropna().unique())
        mz_runs = {i["run_id"] for i in infos}
        if mz_runs and not (psm_runs & mz_runs):
            problems.append(
                "NO RUN NAMES IN COMMON between the mzML files and the PSM "
                f"table. mzML: {sorted(mz_runs)[:4]}; psm.tsv: "
                f"{sorted(psm_runs)[:4]}. Nothing would join, every spectrum "
                "would read as unassigned, and the rescue pile would be "
                "meaningless. Rename the mzML files to match the psm.tsv "
                "Spectrum column, or re-search these exact files.")
        else:
            missing = mz_runs - psm_runs
            if missing:
                problems.append(
                    f"No PSMs for {sorted(missing)}. Those runs will show a "
                    f"0% identification rate, which is correct only if they "
                    f"really were not searched.")
    return infos, problems


def run_pipeline(mzml_paths, psm_paths=None, peptide_path=None,
                 protein_path=None, scorer="rule", model_path=None,
                 qc_threshold=0.6, min_tag=3, keep_polymers=False,
                 max_peaks=150, frag_tol=0.02, convert_dir=None,
                 convert_backend=None, max_qvalue=0.01, assume_prefiltered=False,
                 progress: Callable[[float, str], None] | None = None):
    """
    mzML in, triaged spectrum table out. `progress(fraction, message)` is
    called as work completes so a UI can show something honest.
    """
    def tick(f, msg):
        if progress:
            progress(min(max(f, 0.0), 1.0), msg)

    mzml_paths = list(mzml_paths)
    psm_paths = list(psm_paths or [])
    frames, warnings = [], []

    if any(convert.is_raw(p) for p in mzml_paths):
        cdir = convert_dir or os.path.join(
            os.path.dirname(os.path.abspath(mzml_paths[0])), "msqc_mzml")
        tick(0.02, "Converting RAW files to mzML…")
        mzml_paths, conversions = convert.ensure_mzml(
            mzml_paths, cdir, backend=convert_backend,
            progress=lambda f, m: tick(0.02 + 0.03 * f, m))
        if conversions:
            warnings.append(
                f"Converted {len(conversions)} RAW file(s) to centroided "
                f"mzML in {cdir}. FragPipe must have searched the SAME "
                f"conversion, or scan numbers will not line up with your "
                f"psm.tsv.")
    n = len(mzml_paths)

    for k, path in enumerate(mzml_paths):
        name = os.path.basename(path)
        tick(0.05 + 0.70 * k / n, f"Extracting QC features from {name}…")
        df, warn = extract.extract_run_safe(
            path, keep_peaks=True, max_peaks=max_peaks, frag_tol=frag_tol)
        if warn:
            warnings.append(warn)
            if progress:
                tick(0.05 + 0.70 * k / n, warn)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError(
            "No MS2 spectra could be read from any file. "
            + (" ".join(warnings) if warnings else
               "Check that these really are mzML files with MS2 scans."))

    tick(0.78, "Merging runs…")
    qc = pd.concat(frames, ignore_index=True).sort_values(
        ["run_id", "scan_number"]).reset_index(drop=True)

    if psm_paths:
        tick(0.83, "Joining search results…")
        psms = pd.concat([psm_mod.read_psm(p) for p in psm_paths],
                         ignore_index=True)
        qc = psm_mod.join_psms(qc, psms, max_qvalue=max_qvalue,
                               assume_prefiltered=assume_prefiltered)
    else:
        qc["assigned"] = False

    if peptide_path or protein_path:
        tick(0.87, "Attaching peptide and protein context…")
        try:
            qc = psm_mod.attach_context(qc, peptide_path, protein_path)
        except Exception:
            pass  # context is a nicety; never fail the run over it

    tick(0.90, "Scoring spectrum quality…")
    qc = score.add_quality_score(qc, scorer, model_path)

    tick(0.95, "Assigning triage buckets…")
    qc = rescue.triage(qc, qc_threshold, min_tag,
                       drop_polymers=not keep_polymers)

    tick(1.0, "Done.")
    qc.attrs["warnings"] = warnings
    qc.attrs["mzml_paths"] = mzml_paths
    return qc


def run_summary(qc: pd.DataFrame) -> pd.DataFrame:
    """Per-run QC metrics, the table you watch to catch a bad injection."""
    rows = []
    for run, g in qc.groupby("run_id"):
        rows.append({
            "run_id": run,
            "n_ms2": len(g),
            "id_rate": g["assigned"].mean() if "assigned" in g else float("nan"),
            "rescue_candidates": int(g["is_rescue_candidate"].sum()),
            "median_isolation_purity": g["isolation_purity"].median(),
            "frac_chimeric": (g["isolation_purity"] < 0.5).mean(),
            "frac_polymer": g["is_polymer_like"].mean(),
            "frac_charge_imputed": g["charge_imputed"].mean(),
            "median_longest_tag": g["longest_tag"].median(),
            "median_injection_time_ms": g["injection_time_ms"].median(),
        })
    return pd.DataFrame(rows)
