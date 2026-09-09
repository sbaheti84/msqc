"""
Read search-engine output and join it to the QC feature table on
(run_id, scan_number).

Supported: FragPipe psm.tsv, Sage results.sage.tsv, Casanovo mzTab,
and a generic CSV/TSV with explicit column mapping.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

SPECTRUM_RE = re.compile(r"^(.*?)\.(\d+)\.(\d+)\.(\d+)$")


def _split_fragpipe_spectrum(s: str):
    """'run.00123.00123.2' -> ('run', 123, 2)"""
    m = SPECTRUM_RE.match(str(s).strip())
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), int(m.group(4))


def read_fragpipe_psm(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    cols = {c.lower().strip(): c for c in df.columns}

    spec_col = cols.get("spectrum")
    if spec_col is None:
        raise ValueError(f"{path}: no 'Spectrum' column; is this a psm.tsv?")

    parsed = df[spec_col].map(_split_fragpipe_spectrum)
    out = pd.DataFrame({
        "run_id": [p[0] for p in parsed],
        "scan_number": [p[1] for p in parsed],
    })
    out["peptide"] = df[cols.get("peptide", spec_col)]
    out["modified_peptide"] = df.get(cols.get("modified peptide", ""), out["peptide"])
    from .peptides import fragpipe_sequence
    assigned_mods = df.get(cols.get("assigned modifications", ""), pd.Series("", index=df.index))
    calc_mass = pd.to_numeric(df.get(cols.get("calculated peptide mass", ""), pd.Series(np.nan, index=df.index)), errors="coerce")
    normalized = [fragpipe_sequence(p, m, a, mass) for p,m,a,mass in zip(
        out["peptide"], out["modified_peptide"], assigned_mods, calc_mass)]
    out["modified_peptide_raw"] = out["modified_peptide"]
    out["modified_peptide"] = [p[0] for p in normalized]
    out["annotation_error"] = [p[1] for p in normalized]
    out["protein"] = df.get(cols.get("protein", ""), "")
    out["search_score"] = pd.to_numeric(
        df.get(cols.get("hyperscore", "")), errors="coerce")
    out["delta_score"] = out["search_score"] - pd.to_numeric(
        df.get(cols.get("nextscore", "")), errors="coerce")
    out["expectation"] = pd.to_numeric(
        df.get(cols.get("expectation", "")), errors="coerce")
    out["delta_mass"] = pd.to_numeric(
        df.get(cols.get("delta mass", "")), errors="coerce")
    out["is_decoy"] = out["protein"].astype(str).str.contains("rev_|DECOY", case=False)
    for names, dest in [(("q-value", "q value", "spectrum q-value", "spectrum_q"), "psm_qvalue"),
                        (("posterior error probability", "pep"), "psm_pep"),
                        (("precursor error ppm", "calibrated observed mass error (ppm)"), "precursor_error_ppm")]:
        col = next((cols[n] for n in names if n in cols), None)
        out[dest] = pd.to_numeric(df[col], errors="coerce") if col else np.nan
    out["engine"] = "fragpipe"
    out = out.dropna(subset=["scan_number"])
    out["scan_number"] = out["scan_number"].astype(int)
    return out


def read_sage_psm(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    cols = {c.lower().strip(): c for c in df.columns}
    scan_col = cols.get("scannr") or cols.get("scan")
    scans = df[scan_col].astype(str).str.extract(r"(\d+)$")[0].astype(float)
    filename = df.get(cols.get("filename", ""), "")
    out = pd.DataFrame({
        "run_id": [os.path.splitext(os.path.basename(str(f)))[0] for f in filename]
                  if len(filename) else "",
        "scan_number": scans,
        "peptide": df.get(cols.get("peptide", ""), ""),
        "modified_peptide": df.get(cols.get("peptide", ""), ""),
        "protein": df.get(cols.get("proteins", ""), ""),
        "search_score": pd.to_numeric(df.get(cols.get("hyperscore", "")), errors="coerce"),
        "delta_score": pd.to_numeric(df.get(cols.get("delta_next", "")), errors="coerce"),
        "expectation": pd.to_numeric(df.get(cols.get("spectrum_q", "")), errors="coerce"),
        "delta_mass": pd.to_numeric(df.get(cols.get("delta_mass", "")), errors="coerce"),
        "isotope_error": pd.to_numeric(df.get(cols.get("isotope_error", "")), errors="coerce"),
        "psm_qvalue": pd.to_numeric(df.get(cols.get("spectrum_q", "")), errors="coerce"),
    })
    out["is_decoy"] = df.get(cols.get("label", 1), 1) < 0
    out["engine"] = "sage"
    out = out.dropna(subset=["scan_number"])
    out["scan_number"] = out["scan_number"].astype(int)
    return out


def read_casanovo_mztab(path: str, run_id: str | None = None,
                        manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    """Resolve native scan IDs or MGF indices through an explicit export manifest.

    Keep competing predictions in the raw output; select the best score per
    spectrum here and expose candidate count and runner-up gap.
    """
    from urllib.parse import unquote, urlparse
    rows, header, locations = [], None, {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "MTD" and len(parts) >= 3:
                m = re.fullmatch(r"ms_run\[(\d+)\]-location", parts[1])
                if m:
                    locations[m[1]] = unquote(parts[2]).replace("\\", "/").rsplit("/", 1)[-1]
            elif parts[0] == "PSH":
                header = parts
            elif parts[0] == "PSM" and header:
                if len(parts) != len(header):
                    raise ValueError("Malformed mzTab PSM row.")
                rows.append(dict(zip(header, parts)))
    columns = ["run_id", "scan_number", "denovo_peptide", "denovo_score",
               "denovo_candidate_count", "denovo_score_gap", "denovo_modifications", "denovo_annotation_error"]
    records = []
    for row in rows:
        ref = row.get("spectra_ref", "")
        m = re.fullmatch(r"ms_run\[(\d+)\]:(.*)", ref)
        if not m:
            raise ValueError(f"Unsupported spectra_ref: {ref}")
        filename, native = locations.get(m[1], ""), m[2]
        idx = re.search(r"(?:^|\s)(index|scan)[=:](\d+)$", native)
        if manifest is not None:
            sub = manifest[manifest["mgf_file"] == filename]
            if idx:
                key = "mgf_index" if idx[1] == "index" else "export_scan"
                sub = sub[sub[key] == int(idx[2])]
            else:
                sub = sub[sub["title"] == native]
            if len(sub) != 1:
                raise ValueError(f"Cannot uniquely resolve {filename}: {native} through MGF manifest.")
            run, scan = sub.iloc[0]["run_id"], int(sub.iloc[0]["scan_number"])
        else:
            if not idx or idx[1] == "index" or filename.lower().endswith('.mgf'):
                raise ValueError("MGF/index predictions require an export manifest; an index is not a scan number.")
            run = run_id or re.sub(r"\.(mzML|mzXML|raw|d)$", "", filename, flags=re.I)
            if not run:
                raise ValueError("Prediction has no resolvable run ID.")
            scan = int(idx[2])
        seq = row.get("opt_global_cv_MS:1003169_proforma_peptidoform_sequence")
        if not seq or seq == "null":
            seq = row.get("sequence", "")
        if not seq or seq == "null":
            continue
        records.append(dict(run_id=run, scan_number=scan, denovo_peptide=seq,
                            denovo_score=pd.to_numeric(row.get("search_engine_score[1]"), errors="coerce"),
                            denovo_modifications=row.get("modifications", "null"),
                            denovo_annotation_error=("Exact modified sequence is unavailable; do not annotate the stripped sequence." if
                                row.get("modifications", "null") not in ("null", "0", "") and not re.search(r"[\[+-]", seq) else "")))
    if not records:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(records).sort_values("denovo_score", ascending=False, na_position="last")
    keys = ["run_id", "scan_number"]
    counts = out.groupby(keys).size().rename("denovo_candidate_count")
    gaps = out.groupby(keys)["denovo_score"].apply(
        lambda v: v.iloc[0] - v.iloc[1] if len(v) > 1 else np.nan).rename("denovo_score_gap")
    return out.drop_duplicates(keys).merge(counts, on=keys).merge(gaps, on=keys)


def read_psm(path: str, engine: str = "auto") -> pd.DataFrame:
    if engine == "auto":
        base = os.path.basename(path).lower()
        if "sage" in base:
            engine = "sage"
        elif base.endswith(".mztab"):
            engine = "casanovo"
        else:
            engine = "fragpipe"
    if engine == "fragpipe":
        return read_fragpipe_psm(path)
    if engine == "sage":
        return read_sage_psm(path)
    if engine == "casanovo":
        return read_casanovo_mztab(path)
    raise ValueError(f"unknown engine: {engine}")


def join_psms(qc: pd.DataFrame, psms: pd.DataFrame,
              match_on_run: bool = True, max_qvalue: float = 0.01,
              assume_prefiltered: bool = False, run_mapping: dict | None = None) -> pd.DataFrame:
    """Join exact spectrum keys; only confidence-qualified targets are assigned.

    Missing confidence remains tentative unless the caller explicitly declares
    the input prefiltered. PEP is retained as evidence, not confused with a q-value.
    """
    if not 0 <= max_qvalue <= 1:
        raise ValueError("max_qvalue must be between 0 and 1")
    qc = qc.copy()
    if psms.empty:
        qc["assigned"] = False
        qc["assignment_status"] = "unassigned"
        return qc
    psms = psms.copy()
    if run_mapping:
        psms["run_id"] = psms["run_id"].replace(run_mapping)
    if not match_on_run and (qc["run_id"].nunique() != 1 or psms["run_id"].nunique() != 1):
        raise ValueError("Scan-only matching requires exactly one QC run and one PSM run.")
    keys = ["run_id", "scan_number"] if match_on_run else ["scan_number"]
    if match_on_run and not set(qc["run_id"]) & set(psms["run_id"]):
        raise ValueError("No matching run IDs. Supply an explicit run_mapping; scan-only fallback is disabled.")
    if not match_on_run:
        psms = psms.drop(columns="run_id")
    decoy = psms.get("is_decoy", pd.Series(False, index=psms.index)).fillna(False).astype(bool)
    q = pd.to_numeric(psms.get("psm_qvalue", pd.Series(np.nan, index=psms.index)), errors="coerce")
    psms["_accepted"] = (~decoy & (q.between(0, max_qvalue) | (q.isna() & assume_prefiltered)))
    # Decoys never supply a target annotation; retain tentative target candidates.
    psms = psms[~decoy].copy()
    psms["search_score"] = pd.to_numeric(psms.get("search_score", np.nan), errors="coerce")
    psms = psms.sort_values(["_accepted", "search_score"], ascending=False).drop_duplicates(keys)
    overlap = [c for c in psms.columns if c in qc and c not in keys]
    qc = qc.drop(columns=overlap)
    merged = qc.merge(psms, on=keys, how="left", validate="many_to_one")
    has_peptide = merged["peptide"].fillna("").ne("")
    merged["assigned"] = merged.pop("_accepted").astype("boolean").fillna(False).astype(bool) & has_peptide
    merged["assignment_status"] = np.select(
        [merged["assigned"], has_peptide], ["confident", "tentative"], default="unassigned")
    return merged


# ---------------------------------------------------------------------------
# Cluster file parsing
# ---------------------------------------------------------------------------

_ID_PATTERNS = [
    # falcon on mzML input: mzspec:<collection>:<file>:scan:<n>
    re.compile(r"^mzspec:[^:]*:(?P<run>[^:]+):(?:scan|index):(?P<scan>\d+)"),
    # generic "<anything>:scan:<n>" or "<anything>:index=<n>"
    re.compile(r"^(?P<run>.+?):(?:scan|index)[:=](?P<scan>\d+)"),
    # falcon on MGF input, and msqc's own MGF titles: <run>.<scan>
    re.compile(r"^(?P<run>.+?)\.(?P<scan>\d+)$"),
    # generic controllerType/controllerNumber/scan nativeID
    re.compile(r"(?P<run>[^\s]*?)\s*controllerType=\d+ controllerNumber=\d+ scan=(?P<scan>\d+)"),
    # bare "scan=N" with the run somewhere before it
    re.compile(r"^(?P<run>.*?)[\s:]*(?:scan|index)=(?P<scan>\d+)"),
]


def parse_spectrum_identifier(series: pd.Series) -> pd.DataFrame:
    """
    Turn whatever a clustering tool calls a spectrum into (run_id, scan_number).

    falcon, MaRaCluster, msCRUSH and spectra-cluster all use different
    identifier conventions, and falcon's own convention changes depending on
    whether you fed it mzML or MGF. Rather than guess, try each pattern in
    turn and keep the first that matches.
    """
    s = series.astype(str).str.strip()
    run = pd.Series(pd.NA, index=s.index, dtype=object)
    scan = pd.Series(pd.NA, index=s.index, dtype=object)

    for pattern in _ID_PATTERNS:
        todo = run.isna()
        if not todo.any():
            break
        ex = s[todo].str.extract(pattern)
        if "scan" not in ex.columns:
            continue
        hit = ex["scan"].notna()
        idx = ex.index[hit]
        run.loc[idx] = ex.loc[hit, "run"].values
        scan.loc[idx] = ex.loc[hit, "scan"].values

    run = run.astype(object).where(run.notna(), "")
    # strip directories and extensions so identifiers match the QC run_id
    run = (run.astype(str)
           .str.replace(r"^.*[/\\]", "", regex=True)
           .str.replace(r"\.(mzML|mzXML|mgf|raw|d)$", "", regex=True,
                        case=False))
    return pd.DataFrame({
        "run_id": run,
        "scan_number": pd.to_numeric(scan, errors="coerce"),
    }, index=s.index)


def read_clusters(path: str) -> pd.DataFrame:
    """
    Read a falcon CSV or a MaRaCluster TSV into
    (run_id, scan_number, cluster_id, cluster_size, cluster_n_runs).

    Cluster size and how many runs a cluster spans are the two numbers that
    decide whether an unidentified signal is worth chasing. A cluster of one
    is noise; a tight cluster seen in twelve samples is a real molecule.
    """
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    raw = pd.read_csv(path, sep=sep, comment="#")
    lower = {c.lower().strip(): c for c in raw.columns}

    id_col = next((lower[k] for k in
                   ("identifier", "title", "spectrum", "spectrum_id",
                    "scan", "usi") if k in lower), raw.columns[0])
    cl_col = next((lower[k] for k in
                   ("cluster", "cluster_id", "clusterid", "cluster_idx")
                   if k in lower), raw.columns[-1])

    parsed = parse_spectrum_identifier(raw[id_col])
    parsed["cluster_id"] = raw[cl_col].values
    parsed = parsed.dropna(subset=["scan_number"])
    parsed["scan_number"] = parsed["scan_number"].astype(int)

    # falcon marks unclustered spectra as -1; they carry no evidence
    parsed = parsed[parsed["cluster_id"].astype(str) != "-1"]
    if parsed.empty:
        return parsed.assign(cluster_size=0, cluster_n_runs=0)

    sizes = parsed.groupby("cluster_id").size().rename("cluster_size")
    nruns = (parsed.groupby("cluster_id")["run_id"].nunique()
             .rename("cluster_n_runs"))
    return parsed.merge(sizes, on="cluster_id").merge(nruns, on="cluster_id")


def read_fragpipe_peptide(path: str) -> pd.DataFrame:
    """
    Read FragPipe peptide.tsv. Used for peptide-level context only: how many
    spectra support each peptide, and its q-value.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    c = {k.lower().strip(): k for k in df.columns}
    seq = c.get("peptide") or c.get("peptide sequence")
    if seq is None:
        raise ValueError(f"{path}: no Peptide column; is this a peptide.tsv?")
    out = pd.DataFrame({"peptide": df[seq].astype(str)})
    for src, dst in [("spectral count", "peptide_spectral_count"),
                     ("probability", "peptide_probability"),
                     ("protein", "peptide_protein")]:
        if src in c:
            out[dst] = df[c[src]]
    return out.drop_duplicates("peptide")


def read_fragpipe_protein(path: str) -> pd.DataFrame:
    """
    Read FragPipe protein.tsv.

    The column worth having is the number of distinct peptides per protein.
    A modified peptide that is the ONLY evidence for its protein is a
    one-hit wonder, and one-hit wonders are where false PTM localisations
    and spurious novel proteins concentrate. The app flags them.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    c = {k.lower().strip(): k for k in df.columns}
    pid = c.get("protein id") or c.get("protein")
    if pid is None:
        raise ValueError(f"{path}: no Protein column; is this a protein.tsv?")
    out = pd.DataFrame({"protein_id": df[pid].astype(str)})
    for src, dst in [("unique peptides", "protein_unique_peptides"),
                     ("total peptides", "protein_total_peptides"),
                     ("combined total peptides", "protein_total_peptides"),
                     ("coverage", "protein_coverage"),
                     ("protein probability", "protein_probability"),
                     ("gene", "gene")]:
        if src in c and dst not in out.columns:
            out[dst] = df[c[src]]
    return out.drop_duplicates("protein_id")


def attach_context(qc: pd.DataFrame, peptide_path=None, protein_path=None):
    """
    Join peptide- and protein-level context onto a spectrum table and derive
    the one-hit-wonder flag.
    """
    qc = qc.copy()
    if peptide_path:
        pep = read_fragpipe_peptide(peptide_path)
        qc = qc.merge(pep, on="peptide", how="left")
    if protein_path:
        prot = read_fragpipe_protein(protein_path)
        # psm.tsv Protein is often "sp|P12345|NAME"; match on the accession
        acc = (qc.get("protein", pd.Series("", index=qc.index))
               .astype(str).str.split("|").str[1]
               .fillna(qc.get("protein", "")))
        qc["protein_id"] = np.where(acc.notna() & (acc != ""), acc,
                                    qc.get("protein", ""))
        qc = qc.merge(prot, on="protein_id", how="left")
        if "protein_total_peptides" in qc.columns:
            qc["one_hit_wonder"] = (
                pd.to_numeric(qc["protein_total_peptides"],
                              errors="coerce") <= 1)
    return qc
