"""
Select rescue candidates and hand them off to falcon / Casanovo / open search.

A rescue candidate is a spectrum that is structurally strong, not explained by
the search engine, and not a polymer. The polymer filter matters more than it
sounds: detergent and PEG ladders look excellent by every peptide-agnostic
quality metric and will otherwise fill your GPU queue.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def triage(df: pd.DataFrame, qc_threshold: float = 0.6,
           min_tag: int = 3, drop_polymers: bool = True) -> pd.DataFrame:
    """Add a `triage_class` column. Five mutually exclusive buckets."""
    df = df.copy()
    assigned = df.get("assigned", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    good = df["qc_score"] >= qc_threshold
    tagged = df["longest_tag"].fillna(0) >= min_tag
    polymer = df.get("is_polymer_like", pd.Series(False, index=df.index)).fillna(False).astype(bool)

    if not drop_polymers:
        polymer = pd.Series(False, index=df.index)

    cls = pd.Series("low_quality_unassigned", index=df.index)
    cls[assigned] = "identified"
    cls[~assigned & polymer] = "polymer_contaminant"
    cls[~assigned & ~polymer & good & tagged] = "rescue_candidate"
    cls[~assigned & ~polymer & good & ~tagged] = "structured_unresolved"

    df["triage_class"] = cls
    df["is_rescue_candidate"] = cls == "rescue_candidate"
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["triage_class"].value_counts()
    total = len(df)
    out = counts.rename("n_spectra").to_frame()
    out["percent"] = (out["n_spectra"] / total * 100).round(2)
    return out.reset_index(names="triage_class")


def write_mgf(df: pd.DataFrame, path: str, title_prefix: str = "") -> int:
    """
    Write rescue candidates as MGF. This is the input format falcon and
    Casanovo both accept, so the same file feeds clustering and de novo.

    Requires the QC table to have been built with --keep-peaks.
    """
    from .identity import export_mgf
    if title_prefix:
        raise ValueError("Title prefixes are no longer supported; spectrum identity is recorded in the manifest.")
    return len(export_mgf(df, path))


def next_step_commands(mgf_path: str, out_dir: str) -> str:
    """Print the exact commands for the next stage. No wrappers, no magic."""
    mgf = os.path.abspath(mgf_path)
    out = os.path.abspath(out_dir)
    return f"""
Next steps for the rescue pile
------------------------------
1. Cluster across runs first. This collapses the GPU workload and gives you
   the reproducibility evidence you will need later.

     pip install falcon-ms
     falcon "{mgf}" {out}/falcon \\
         --export_representatives --precursor_tol 20 ppm --fragment_tol 0.05 \\
         --distance_threshold 0.1

2. For representative sequencing with automatic identity mapping, use the
   Streamlit Run rescue tab. To sequence the original exported queue:

     pip install casanovo
     casanovo sequence --output_dir "{out}" --output_root casanovo "{mgf}"

3. Feed the same MGF to an open search to catch unknown modifications.
   This is cheaper than de novo and usually higher yield.

     (MSFragger: set precursor_mass_lower=-150, precursor_mass_upper=500,
      then run PTM-Shepherd on the result)

4. Bring results back in:

     msqc annotate --qc {out}/qc_triaged.parquet \\
         --casanovo {out}/casanovo.mztab \\
         --manifest "{os.path.splitext(mgf)[0]}.manifest.csv" \\
         --clusters {out}/falcon.csv \\
         --out {out}/qc_annotated.parquet
"""


def validate_psms(qc: "pd.DataFrame", analyzer: str = "orbitrap_hcd",
                  label: str | None = None) -> "pd.DataFrame":
    """
    Run the interpretation checklist over every assigned PSM in a QC table.

    Requires the table to have been extracted with --keep-peaks, and to carry
    a peptide column from `msqc triage`.
    """
    import numpy as np
    import pandas as pd

    from . import checklist as ck

    prec_ppm, frag_ppm, frag_da = ck.ANALYZER_TOLERANCE[analyzer]
    rows = []
    sub = qc[qc.get("assigned", pd.Series(False, index=qc.index)) &
             qc.get("peptide").notna()] if "peptide" in qc else qc.iloc[0:0]

    for _, r in sub.iterrows():
        mz = np.asarray(r["_mz"] if r["_mz"] is not None else [], float)
        inten = np.asarray(r["_intensity"] if r["_intensity"] is not None
                           else [], float)
        annotation_error = r.get("annotation_error")
        if isinstance(annotation_error, str) and annotation_error:
            rows.append({"run_id": r["run_id"], "scan_number": r["scan_number"],
                         "peptide": r["peptide"], "verdict": "warn", "concerns": annotation_error})
            continue
        pep = r.get("modified_peptide")
        if not isinstance(pep, str) or not pep:
            pep = r["peptide"]
        if mz.size == 0 or not isinstance(pep, str) or not pep:
            continue

        try:
            ann = ck.annotate_spectrum(mz, inten, pep, int(r.get("charge", 2) or 2),
                                       frag_ppm=frag_ppm or 20.0, frag_da=frag_da)
        except ValueError as exc:
            rows.append({"run_id": r["run_id"], "scan_number": r["scan_number"],
                         "peptide": pep, "verdict": "warn", "concerns": str(exc)})
            continue
        rec = {k: v for k, v in ann.items()
               if k not in ("matched", "unmatched_mz", "unmatched_intensity")}
        rec.update({"run_id": r["run_id"], "scan_number": r["scan_number"],
                    "peptide": pep, "charge": r.get("charge"),
                    "isolation_purity": r.get("isolation_purity")})
        rec.update(ck.check_dominant_unannotated(ann, inten))
        rec.update(ck.check_ptm_signature_ions(mz, inten, pep))
        rec.update(ck.check_immonium(mz, pep))
        rec.update(ck.check_cleavage_chemistry(pep, ann))

        dm = r.get("delta_mass")
        rec["is_modified"] = bool(dm is not None and np.isfinite(dm)
                                  and abs(dm) > 0.01)
        rec.update(ck.check_delta_mass_ambiguity(dm))

        if label:
            rec.update(ck.check_reporter_ions(mz, inten, label))

        # precursor mass error, if the search reported a theoretical mass
        rec["precursor_error_ppm"] = r.get("precursor_error_ppm", np.nan)

        rec["verdict"], rec["concerns"] = ck.verdict(rec, analyzer)
        rows.append(rec)

    out = pd.DataFrame(rows)
    if not out.empty:
        for name in ("bond_coverage", "longest_consecutive_series", "explained_tic_frac", "median_abs_error_ppm"):
            if name not in out:
                out[name] = np.nan
    return out
