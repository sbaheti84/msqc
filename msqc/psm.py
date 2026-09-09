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
    out["protein"] = df.get(cols.get("protein", ""), "")
    out["search_score"] = pd.to_numeric(
        df.get(cols.get("hyperscore", "")), errors="coerce")
    out["delta_score"] = pd.to_numeric(
        df.get(cols.get("nextscore", "")), errors="coerce")
    out["expectation"] = pd.to_numeric(
        df.get(cols.get("expectation", "")), errors="coerce")
    out["delta_mass"] = pd.to_numeric(
        df.get(cols.get("delta mass", "")), errors="coerce")
    out["is_decoy"] = out["protein"].astype(str).str.contains("rev_|DECOY", case=False)
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
        "delta_mass": pd.to_numeric(df.get(cols.get("isotope_error", "")), errors="coerce"),
    })
    out["is_decoy"] = df.get(cols.get("label", 1), 1) < 0
    out["engine"] = "sage"
    out = out.dropna(subset=["scan_number"])
    out["scan_number"] = out["scan_number"].astype(int)
    return out


def read_casanovo_mztab(path: str, run_id: str | None = None) -> pd.DataFrame:
    """
    Parse the PSM section of a Casanovo mzTab file.

    Casanovo writes spectra_ref as `ms_run[N]:scan=M`, where N indexes the
    `ms_run[N]-location` lines in the metadata block. Those locations must be
    resolved back to run names, otherwise a multi-run mzTab merges on scan
    number alone and silently duplicates rows wherever two runs share a scan
    number - which they almost always do.
    """
    rows, header = [], None
    locations = {}
    loc_re = re.compile(r"ms_run\[(\d+)\]-location")
    with open(path) as fh:
        for line in fh:
            if line.startswith("MTD"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    m = loc_re.search(parts[1])
                    if m:
                        name = parts[2].split("/")[-1].split("\\")[-1]
                        name = re.sub(r"\.(mzML|mzXML|mgf|raw|d)$", "", name,
                                      flags=re.I)
                        locations[m.group(1)] = name
            elif line.startswith("PSH"):
                header = line.rstrip("\n").split("\t")
            elif line.startswith("PSM") and header:
                rows.append(line.rstrip("\n").split("\t"))

    cols = ["run_id", "scan_number", "denovo_peptide", "denovo_score"]
    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows, columns=header)
    ref = df.get("spectra_ref", pd.Series([""] * len(df))).astype(str)

    scans = ref.str.extract(r"scan[=:](\d+)")[0]
    if scans.isna().all():
        scans = ref.str.extract(r"index[=:](\d+)")[0]

    run_idx = ref.str.extract(r"ms_run\[(\d+)\]")[0]
    if run_id:
        runs = pd.Series(run_id, index=df.index)
    elif locations and run_idx.notna().any():
        runs = run_idx.map(locations).fillna("")
    else:
        # No metadata to resolve; try the USI-style forms instead.
        runs = parse_spectrum_identifier(ref)["run_id"]

    out = pd.DataFrame({
        "run_id": runs.fillna("").values,
        "scan_number": pd.to_numeric(scans, errors="coerce").values,
        "denovo_peptide": df.get("sequence", "").values,
        "denovo_score": pd.to_numeric(df.get("search_engine_score[1]"),
                                      errors="coerce").values,
    }).dropna(subset=["scan_number"])
    out["scan_number"] = out["scan_number"].astype(int)
    return out


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
              match_on_run: bool = True) -> pd.DataFrame:
    """
    Left-join PSMs onto the QC table. If run IDs do not line up between the
    mzML filename and the search output, fall back to scan number only and
    say so loudly, because silent mismatches produce a fake rescue pile.
    """
    if psms.empty:
        qc = qc.copy()
        qc["assigned"] = False
        return qc

    psms = psms[~psms.get("is_decoy", False).astype(bool)].copy()

    keys = ["run_id", "scan_number"] if match_on_run else ["scan_number"]
    if match_on_run:
        overlap = set(qc["run_id"]) & set(psms["run_id"])
        if not overlap:
            print("  [warn] no run_id overlap between mzML and PSM table "
                  f"(mzML: {sorted(set(qc['run_id']))[:3]}, "
                  f"PSM: {sorted(set(psms['run_id']))[:3]}). "
                  "Falling back to scan-number-only join. Verify this is correct.")
            keys = ["scan_number"]
            psms = psms.drop(columns=["run_id"])

    psms = psms.sort_values("search_score", ascending=False).drop_duplicates(keys)
    merged = qc.merge(psms, on=keys, how="left", suffixes=("", "_psm"))
    merged["assigned"] = merged["peptide"].notna()

    rate = merged["assigned"].mean()
    print(f"  joined {int(merged['assigned'].sum()):,} PSMs to "
          f"{len(merged):,} MS2 scans ({rate:.1%} identification rate)")
    if rate < 0.02:
        print("  [warn] identification rate under 2%. The join key is probably "
              "wrong, or the search failed. Check before trusting rescue output.")
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
