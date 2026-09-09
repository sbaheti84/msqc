"""
msqc Streamlit app.

Run with:
    streamlit run app.py -- --qc results/qc_triaged.parquet

or launch bare and point it at a file from the sidebar:
    streamlit run app.py

Scale note. Streamlit reruns the whole script on every widget interaction, so
everything expensive is behind st.cache_data keyed on the file path and its
modification time. Comfortable to roughly 200k spectra on a laptop. Past that,
the tables stay the same but you will want the queries pushed into DuckDB
rather than held in a dataframe.
"""

from __future__ import annotations

import glob
import io
import os
import sys
import tempfile
import uuid
import hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from msqc import checklist as ck
from msqc import convert, fragments, pipeline, rescue, score
from msqc.rescue_ui import page_rescue

st.set_page_config(page_title="msqc", layout="wide",
                   initial_sidebar_state="expanded")

BUCKET_COLOURS = {
    "identified": "#5B7FA6",
    "rescue_candidate": "#D08A2E",
    "structured_unresolved": "#7C6BA8",
    "polymer_contaminant": "#A6564E",
    "low_quality_unassigned": "#9AA3AA",
}

FEATURE_HELP = {
    "isolation_purity": "Fraction of intensity in the isolation window that "
                        "belongs to the target precursor. Below 0.5 the MS2 "
                        "is a mixture and no engine will explain it with one "
                        "peptide. The strongest chimera indicator there is.",
    "longest_tag": "Longest chain of peaks whose m/z gaps match amino acid "
                   "residue masses. The best single indicator that a spectrum "
                   "contains sequenceable peptide fragmentation.",
    "n_complementary": "Peak pairs summing to the precursor neutral mass plus "
                       "two protons, i.e. matching b/y partners.",
    "entropy": "Shannon entropy of normalised intensities. High entropy with "
               "no structure means noise.",
    "snr_proxy": "Base peak over the estimated noise floor.",
    "norm_entropy": "Entropy divided by log(n peaks). Near 1.0 means a flat, "
                    "structureless spectrum.",
    "ladder_length": "Longest evenly spaced peak ladder. Polymers and "
                     "detergents produce long ones; peptides do not.",
    "n_cofragmented": "Distinct precursor envelopes sharing the isolation "
                      "window.",
    "gap_density": "Residue-mass gaps per peak.",
    "complementary_tic_frac": "Fraction of intensity sitting on b/y "
                              "complementary pairs.",
    "isotope_tic_frac": "Fraction of intensity in resolved isotope clusters. "
                        "Low values mean electronic or chemical noise.",
}


# ---------------------------------------------------------------------------
# Loading and running
# ---------------------------------------------------------------------------

if "workdir" not in st.session_state:
    root = os.environ.get("MSQC_WORKDIR")
    if root:
        os.makedirs(root, exist_ok=True)
    st.session_state["workdir"] = tempfile.mkdtemp(prefix="msqc_", dir=root)
WORKDIR = st.session_state["workdir"]


def _save_uploads(files, subdir):
    """Stream uploads to disk. mzML files are far too big to hold in memory."""
    d = os.path.join(WORKDIR, subdir)
    os.makedirs(d, exist_ok=True)
    paths = []
    for f in files or []:
        dest = os.path.join(d, os.path.basename(f.name))
        with open(dest, "wb") as fh:
            fh.write(f.getbuffer())
        paths.append(dest)
    return paths


def _expand(patterns):
    out = []
    for pat in patterns:
        pat = pat.strip()
        if not pat:
            continue
        hits = sorted(glob.glob(pat))
        out.extend(hits if hits else ([pat] if os.path.exists(pat) else []))
    return out


@st.cache_data(show_spinner=False)
def cached_pipeline(mzml_paths, psm_paths, peptide_path, protein_path,
                    scorer, qc_threshold, min_tag, keep_polymers, max_peaks,
                    _progress=None):
    return pipeline.run_pipeline(
        list(mzml_paths), list(psm_paths), peptide_path, protein_path,
        scorer=scorer, qc_threshold=qc_threshold, min_tag=min_tag,
        keep_polymers=keep_polymers, max_peaks=max_peaks, progress=_progress)


@st.cache_data(show_spinner="Reading QC table…")
def load_parquet(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Running the interpretation checklist…")
def run_checklist_cached(df: pd.DataFrame, analyzer: str, label):
    return rescue.validate_psms(df, analyzer=analyzer, label=label)


def _cli_default():
    for i, arg in enumerate(sys.argv):
        if arg == "--qc" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--qc="):
            return arg.split("=", 1)[1]
    return ""


def page_load():
    """Tab 1. Upload or point at files, preflight, then run."""
    st.subheader("Load data")

    mode = st.radio(
        "How are the files reaching this app?",
        ["Upload files", "Paths on this machine", "Existing qc_triaged.parquet"],
        horizontal=True,
        help="Uploading is easiest but goes through browser memory. For mzML "
             "over ~1 GB use paths instead, and raise "
             "server.maxUploadSize if you must upload.", key="radio_130")

    mzml_paths, psm_paths, pep_path, prot_path = [], [], None, None

    if mode == "Upload files":
        st.caption("mzML files can be large. Streamlit's default upload cap "
                   "is 200 MB per file; raise it with "
                   "`--server.maxUploadSize 4000` or use the paths option.")
        c = st.columns(2)
        with c[0]:
            up_mz = st.file_uploader(
                "mzML or Thermo RAW files",
                type=["mzML", "mzml", "raw", "RAW"],
                accept_multiple_files=True, key="file_uploader_146")
        with c[1]:
            up_psm = st.file_uploader("psm.tsv (one per run, or combined)",
                                      type=["tsv", "txt"],
                                      accept_multiple_files=True, key="file_uploader_151")
            up_pep = st.file_uploader("peptide.tsv (optional)",
                                      type=["tsv", "txt"], key="file_uploader_154")
            up_prot = st.file_uploader("protein.tsv (optional)",
                                       type=["tsv", "txt"], key="file_uploader_156")
        if up_mz:
            mzml_paths = _save_uploads(up_mz, "mzml")
        if up_psm:
            psm_paths = _save_uploads(up_psm, "psm")
        if up_pep:
            pep_path = _save_uploads([up_pep], "psm")[0]
        if up_prot:
            prot_path = _save_uploads([up_prot], "psm")[0]

    elif mode == "Paths on this machine":
        c = st.columns(2)
        mz_txt = c[0].text_area(
            "mzML paths or globs, one per line", height=120,
            placeholder="/data/hela/*.raw   or   /data/hela/*.mzML", key="text_area_169")
        psm_txt = c[1].text_area(
            "psm.tsv paths or globs, one per line", height=120,
            placeholder="/data/hela/fragpipe/psm.tsv", key="text_area_172")
        c2 = st.columns(2)
        pep_in = c2[0].text_input("peptide.tsv (optional)", key="text_input_176")
        prot_in = c2[1].text_input("protein.tsv (optional)", key="text_input_177")
        mzml_paths = _expand(mz_txt.splitlines())
        psm_paths = _expand(psm_txt.splitlines())
        pep_path = pep_in if pep_in and os.path.exists(pep_in) else None
        prot_path = prot_in if prot_in and os.path.exists(prot_in) else None

    else:
        c = st.columns([3, 1])
        p = c[0].text_input("Parquet path", value=_cli_default(),
                            placeholder="results/qc_triaged.parquet", key="text_input_185")
        up = st.file_uploader("…or upload it", type=["parquet"], key="file_uploader_187")
        if up is not None:
            source_key = (up.name, hashlib.sha256(up.getvalue()).hexdigest())
            if st.session_state.get("qc_source") != source_key:
                st.session_state["qc"] = pd.read_parquet(io.BytesIO(up.getvalue()))
                st.session_state["qc_key"] = str(uuid.uuid4())
                st.session_state["qc_source"] = source_key
            st.success(f"Loaded {len(st.session_state['qc']):,} spectra.")
        elif p and os.path.exists(p):
            source_key = (p, os.path.getmtime(p))
            if st.session_state.get("qc_source") != source_key:
                st.session_state["qc"] = load_parquet(p, os.path.getmtime(p))
                st.session_state["qc_key"] = str(uuid.uuid4())
                st.session_state["qc_source"] = source_key
            st.success(f"Loaded {len(st.session_state['qc']):,} spectra.")
        elif p:
            st.error("File not found.")
        return

    if not mzml_paths:
        st.info("Add at least one mzML file to continue.")
        return

    st.markdown(f"**{len(mzml_paths)} mzML** and **{len(psm_paths)} PSM "
                f"table(s)** ready.")
    if not psm_paths:
        st.warning("Without a psm.tsv every spectrum is treated as "
                   "unassigned, so the rescue pile will be meaningless. "
                   "Only run this way if you genuinely have no search "
                   "results.")

    # ---- RAW conversion ---------------------------------------------------
    raws = [p for p in mzml_paths if convert.is_raw(p)]
    conv_backend = None
    if raws:
        st.divider()
        st.markdown("### RAW conversion")
        backends = convert.available_backends()
        usable = [b for b in backends if b["ok"]]
        best = convert.best_backend()

        if not usable:
            st.error(
                f"**{len(raws)} Thermo RAW file(s), but no converter on this "
                f"machine.** Nothing in Python reads Thermo RAW directly. "
                f"Install one of these, then reload:")
            for b in backends:
                st.markdown(f"- **{b['label']}** — `{b['install']}`")
            st.stop()

        conv_backend = st.selectbox(
            "Converter", [b["id"] for b in usable],
            index=[b["id"] for b in usable].index(best) if best else 0,
            format_func=lambda i: next(b["label"] for b in backends
                                       if b["id"] == i), key="selectbox_231")
        st.info(
            f"{len(raws)} RAW file(s) will be converted to **centroided, "
            f"indexed mzML**. This is the slowest step in the pipeline. "
            f"Results are cached next to the source files and reused, so you "
            f"pay it once.")
        st.warning(
            "**Your psm.tsv must come from the same conversion.** If "
            "FragPipe searched a differently-converted mzML, scan numbers "
            "will not line up and nothing will join. Preflight checks the "
            "run names, but it cannot detect two conversions of the same "
            "run that number scans differently.")

    # ---- preflight --------------------------------------------------------
    st.divider()
    st.markdown("### Preflight")
    st.caption("Checks the mzML is usable and, critically, that the run names "
               "in psm.tsv match the mzML filenames. A mismatch joins nothing "
               "and silently reports a 0% identification rate.")

    with st.spinner("Inspecting files…"):
        infos, problems = pipeline.preflight(mzml_paths, psm_paths)

    if not infos and raws:
        st.caption("RAW files are inspected after conversion; only the "
                   "converter check applies here.")
    if infos:
        t = pd.DataFrame(infos)[
            [c for c in ["run_id", "n_ms1", "n_ms2", "centroided",
                         "truncated", "median_peaks", "scan_min", "scan_max",
                         "n_charge_missing", "n_no_isolation"]
             if c in pd.DataFrame(infos).columns]]
        st.dataframe(t, use_container_width=True, hide_index=True)
        st.caption("Counts come from a probe of the first few hundred "
                   "spectra, not the whole file.")

    blocking = [p for p in problems if "NO RUN NAMES" in p
                or "no MS2" in p or "could not be read" in p
                or "profile-mode" in p]
    truncated = [p for p in problems if "TRUNCATED" in p]
    if truncated:
        st.error("**One or more mzML files are incomplete.** The pipeline "
                 "will salvage the spectra that parse, but every per-run "
                 "rate below will be computed over a partial gradient. "
                 "Re-transfer or re-convert before you trust the numbers.")
    for p in problems:
        (st.error if p in blocking else st.warning)(p)
    if not problems:
        st.success("No problems found.")
    if blocking:
        st.stop()

    # ---- settings and run -------------------------------------------------
    st.divider()
    st.markdown("### Run")
    c = st.columns(4)
    scorer = c[0].selectbox("Quality scorer", ["rule", "model"],
                            help="Start with 'rule'. It needs no training "
                                 "data and works on the first file you "
                                 "process.", key="selectbox_291")
    model_path = st.text_input("Trained QC model path", key="load_model_path") if scorer == "model" else None
    qthr = c[1].slider("Quality threshold", 0.0, 1.0, 0.6, 0.05, key="slider_295")
    mtag = c[2].slider("Minimum sequence tag", 0, 8, 3, key="slider_296")
    maxp = c[3].number_input("Peaks per spectrum", 50, 500, 150, 25,
                             help="Lower this if extraction is slow. The "
                                  "pairwise gap matrix is quadratic in this "
                                  "number.", key="number_input_297")
    keep_poly = st.checkbox("Keep polymer-like spectra in the rescue queue",
                            value=False,
                            help="Leave off. A PEG ladder scores well on "
                                 "every peptide-agnostic metric and will "
                                 "otherwise fill your GPU queue with "
                                 "detergent.", key="checkbox_301")

    confidence = st.columns(2)
    max_qvalue = confidence[0].number_input("Maximum input PSM q-value", min_value=0., max_value=1., value=.01, format="%.4f", key="load_max_qvalue")
    assume_prefiltered = confidence[1].checkbox("Input PSM tables are already FDR-filtered", key="load_prefiltered",
        help="Explicitly allow target PSMs without q-values to count as identified. Otherwise they remain tentative.")

    est = sum(os.path.getsize(p) for p in mzml_paths) / 1e6
    st.caption(f"About {est:,.0f} MB of mzML. Extraction runs at roughly "
               f"300-400 spectra per second per core.")

    if st.button("Run QC pipeline", type="primary", use_container_width=True, key="button_312"):
        bar = st.progress(0.0, text="Starting…")

        def cb(frac, msg):
            bar.progress(frac, text=msg)

        try:
            qc = pipeline.run_pipeline(
                mzml_paths, psm_paths, pep_path, prot_path,
                scorer=scorer, model_path=model_path, qc_threshold=qthr, min_tag=mtag,
                keep_polymers=keep_poly, max_peaks=int(maxp),
                convert_backend=conv_backend, max_qvalue=max_qvalue,
                assume_prefiltered=assume_prefiltered, progress=cb)
        except Exception as e:
            bar.empty()
            st.exception(e)
            return
        bar.empty()

        for w in qc.attrs.get("warnings", []):
            st.error(w)
        st.session_state["qc"] = qc
        st.session_state["qc_key"] = str(uuid.uuid4())
        st.session_state["mzml_paths"] = qc.attrs.get("mzml_paths", mzml_paths)
        st.success(f"Done. {len(qc):,} MS2 spectra across "
                   f"{qc['run_id'].nunique()} run(s).")
        st.dataframe(pipeline.run_summary(qc).round(3),
                     use_container_width=True, hide_index=True)

        out = os.path.join(WORKDIR, "qc_triaged.parquet")
        qc.to_parquet(out, index=False)
        with open(out, "rb") as fh:
            st.download_button("Download qc_triaged.parquet", fh.read(),
                               "qc_triaged.parquet", key="download_button_342")
        st.info("Open Run rescue to cluster spectra and run external search or de novo tools from this app.")


# ---------------------------------------------------------------------------
# Shared widgets
# ---------------------------------------------------------------------------

def sidebar_filters(df):
    st.sidebar.header("Filters")
    f = pd.Series(True, index=df.index)

    runs = sorted(df["run_id"].dropna().unique())
    if len(runs) > 1:
        sel = st.sidebar.multiselect("Runs", runs, default=runs, key="multiselect_359")
        f &= df["run_id"].isin(sel)

    buckets = [b for b in BUCKET_COLOURS if b in set(df["triage_class"])]
    selb = st.sidebar.multiselect("Triage class", buckets, default=buckets, key="multiselect_363")
    f &= df["triage_class"].isin(selb)

    lo, hi = st.sidebar.slider("Quality score", 0.0, 1.0, (0.0, 1.0), 0.01, key="slider_366")
    f &= df["qc_score"].between(lo, hi)

    if df["charge"].notna().any():
        cz = sorted(int(c) for c in df["charge"].dropna().unique())
        selz = st.sidebar.multiselect("Charge", cz, default=cz, key="multiselect_371")
        f &= df["charge"].isin(selz)

    rt = df["rt_min"]
    if rt.notna().any():
        r0, r1 = float(rt.min()), float(rt.max())
        if r1 > r0:
            a, b = st.sidebar.slider("Retention time (min)", r0, r1, (r0, r1), key="slider_378")
            f &= rt.between(a, b)

    tag = st.sidebar.slider("Minimum sequence tag", 0,
                            max(int(df["longest_tag"].fillna(0).max()), 1), 0, key="slider_381")
    f &= df["longest_tag"] >= tag

    if df["isolation_purity"].notna().any():
        p = st.sidebar.slider("Minimum isolation purity", 0.0, 1.0, 0.0, 0.05, key="slider_386")
        f &= df["isolation_purity"].fillna(1.0) >= p

    st.sidebar.caption(f"{int(f.sum()):,} of {len(df):,} spectra selected")
    return df[f]


def mirror_plot(mz, inten, peptide, charge, frag_tol=0.02, title=""):
    """Observed spectrum above, matched theoretical fragments below."""
    mz = np.asarray(mz, float)
    inten = np.asarray(inten, float)
    if inten.size == 0:
        return go.Figure()
    rel = 100 * inten / inten.max()

    fig = go.Figure()
    matched_idx, ann = set(), []
    if peptide:
        theo = fragments.theoretical_fragments(
            peptide, max_charge=max(int(charge or 2) - 1, 1))
        for t in theo:
            j = int(np.argmin(np.abs(mz - t["mz"])))
            if abs(mz[j] - t["mz"]) <= frag_tol:
                matched_idx.add(j)
                ann.append((t, j))

    # observed, unmatched
    um = [i for i in range(mz.size) if i not in matched_idx]
    fig.add_trace(go.Bar(x=mz[um], y=rel[um], width=0.9, name="unmatched",
                         marker_color="#B8BFC4",
                         hovertemplate="m/z %{x:.4f}<br>%{y:.1f}%<extra></extra>"))
    if matched_idx:
        mi = sorted(matched_idx)
        labels = {j: t["label"] for t, j in ann}
        colours = ["#3D6FA8" if labels[j].startswith("b") else "#B5432F"
                   for j in mi]
        fig.add_trace(go.Bar(
            x=mz[mi], y=rel[mi], width=0.9, name="matched b/y",
            marker_color=colours,
            text=[labels[j] for j in mi], textposition="outside",
            hovertemplate="%{text}<br>m/z %{x:.4f}<br>%{y:.1f}%<extra></extra>"))
        # theoretical, mirrored below the axis
        fig.add_trace(go.Bar(
            x=[t["mz"] for t, _ in ann],
            y=[-rel[j] for _, j in ann], width=0.9, name="theoretical",
            marker_color=["#3D6FA8" if t["label"].startswith("b") else "#B5432F"
                          for t, _ in ann],
            opacity=0.45,
            customdata=[t["label"] for t, _ in ann],
            hovertemplate="%{customdata}<br>theoretical m/z %{x:.4f}"
                          "<extra></extra>"))

    fig.add_hline(y=0, line_width=1, line_color="#444")
    fig.update_layout(
        title=title, barmode="overlay", bargap=0,
        height=430, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="m/z", yaxis_title="relative intensity (%)",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        plot_bgcolor="white")
    fig.update_yaxes(zeroline=True, gridcolor="#EEE")
    fig.update_xaxes(gridcolor="#F5F5F5")
    return fig


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview(df, view):
    st.subheader("Run overview")

    c = st.columns(5)
    c[0].metric("MS2 spectra", f"{len(df):,}")
    idr = df["assigned"].mean() if "assigned" in df else np.nan
    c[1].metric("Identification rate", f"{idr:.1%}")
    c[2].metric("Rescue candidates", f"{int(df['is_rescue_candidate'].sum()):,}")
    pur = df["isolation_purity"].median()
    c[3].metric("Median isolation purity",
                "n/a" if pd.isna(pur) else f"{pur:.2f}",
                help="Below 0.7 means co-isolation is your dominant problem, "
                     "not spectrum quality.")
    c[4].metric("Polymer-like", f"{df['is_polymer_like'].mean():.1%}",
                help="Above 10% is a sample prep problem.")

    warns = []
    if pd.notna(pur) and pur < 0.7:
        warns.append("Median isolation purity is below 0.7. Narrow the "
                     "isolation window before chasing spectrum quality.")
    if df["charge_imputed"].mean() > 0.15:
        warns.append(f"{df['charge_imputed'].mean():.0%} of scans have no "
                     "charge state in the mzML. Every mass-dependent feature "
                     "below is guessing on those.")
    if df["is_polymer_like"].mean() > 0.10:
        warns.append("More than 10% of spectra look like polymer ladders.")
    for w in warns:
        st.warning(w)

    left, right = st.columns([3, 2])
    with left:
        counts = (df["triage_class"].value_counts()
                  .rename_axis("triage_class").reset_index(name="n"))
        counts["percent"] = 100 * counts["n"] / len(df)
        fig = px.bar(counts, x="n", y="triage_class", orientation="h",
                     color="triage_class", color_discrete_map=BUCKET_COLOURS,
                     text=counts["percent"].map("{:.1f}%".format))
        fig.update_layout(showlegend=False, height=300, plot_bgcolor="white",
                          yaxis_title="", xaxis_title="spectra",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**What each bucket means**")
        st.markdown("""
- **identified** — the search explained it. Nothing to do.
- **rescue_candidate** — strong peptide structure, unassigned, not a polymer.
  This is the GPU queue.
- **structured_unresolved** — good signal with insufficient tag evidence.
  Review for alternative fragmentation or specialized search.
- **polymer_contaminant** — repeat ladder. Fix sample prep.
- **low_quality_unassigned** — not enough signal. Most of the pile.
""")

    st.divider()
    st.markdown("**Identification rate across the gradient**")
    bins = st.slider("RT bins", 10, 100, 40, key="rtbins")
    tmp = df.copy()
    tmp["rt_bin"] = pd.cut(tmp["rt_min"], bins)
    g = tmp.groupby("rt_bin", observed=True).agg(
        rt=("rt_min", "mean"), n=("scan_number", "size"),
        id_rate=("assigned", "mean"),
        purity=("isolation_purity", "median")).reset_index(drop=True)
    fig = go.Figure()
    fig.add_bar(x=g["rt"], y=g["n"], name="MS2 scans",
                marker_color="#DDE3E8", yaxis="y2")
    fig.add_scatter(x=g["rt"], y=g["id_rate"], name="identification rate",
                    mode="lines+markers", line_color="#3D6FA8")
    fig.add_scatter(x=g["rt"], y=g["purity"], name="median isolation purity",
                    mode="lines", line=dict(color="#D08A2E", dash="dot"))
    fig.update_layout(height=330, plot_bgcolor="white",
                      xaxis_title="retention time (min)",
                      yaxis=dict(title="rate", range=[0, 1]),
                      yaxis2=dict(overlaying="y", side="right",
                                  title="scans", showgrid=False),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A collapse in identification rate that tracks a collapse in "
               "isolation purity is a co-isolation problem, not a "
               "chromatography problem.")


def page_triage(df, view):
    st.subheader("Triage quadrant")
    st.caption("Quality score against search score. The high-quality, "
               "unassigned corner is the rescue pile.")

    c = st.columns(4)
    xcol = c[0].selectbox("X axis", ["qc_score", "longest_tag", "entropy",
                                     "isolation_purity", "snr_proxy",
                                     "n_complementary", "rt_min"], index=0, key="selectbox_542")
    ycol = c[1].selectbox("Y axis", [name for name in ["search_score", "qc_score",
                                     "n_peaks_above_noise", "precursor_mz",
                                     "isolation_purity", "longest_tag"] if name in df.columns],
                          index=0, key="selectbox_545")
    n_max = c[2].number_input("Max points", 1000, 200000, 20000, 1000,
                              help="Plotly slows down past ~50k points. "
                                   "Sampling is random and does not change "
                                   "the counts above.", key="number_input_549")
    logy = c[3].checkbox("Log Y", value=False, key="checkbox_553")

    plot = view if len(view) <= n_max else view.sample(int(n_max), random_state=0)
    if len(plot) < len(view):
        st.caption(f"Showing a random {len(plot):,} of {len(view):,}.")

    hover = ["run_id", "scan_number", "precursor_mz", "charge", "longest_tag",
             "isolation_purity", "peptide"]
    fig = px.scatter(plot, x=xcol, y=ycol, color="triage_class",
                     color_discrete_map=BUCKET_COLOURS,
                     hover_data=[h for h in hover if h in plot.columns],
                     opacity=0.55)
    fig.update_traces(marker=dict(size=5, line=dict(width=0)))
    fig.update_layout(height=560, plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"),
                      margin=dict(l=10, r=10, t=10, b=10))
    if logy:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Threshold explorer**")
    st.caption("Re-run triage at different cut-offs to see how the queue size "
               "responds before committing GPU time.")
    c = st.columns(3)
    q = c[0].slider("Quality threshold", 0.0, 1.0, 0.6, 0.05, key="slider_578")
    t = c[1].slider("Minimum tag length", 0, 8, 3, key="slider_579")
    keep_poly = c[2].checkbox("Keep polymers in the queue", value=False, key="checkbox_580")
    re_df = rescue.triage(df.copy(), qc_threshold=q, min_tag=t,
                          drop_polymers=not keep_poly)
    s = rescue.summarise(re_df)
    c2 = st.columns(2)
    c2[0].dataframe(s, use_container_width=True, hide_index=True)
    n_now = int(re_df["is_rescue_candidate"].sum())
    n_before = int(df["is_rescue_candidate"].sum())
    c2[1].metric("Rescue queue at these settings", f"{n_now:,}",
                 delta=f"{n_now - n_before:+,} vs current")
    c2[1].caption("Casanovo on an L4 runs roughly 30-60 spectra per second. "
                  "Cluster first and you sequence representatives, not this "
                  "whole number.")


def page_spectrum(df, view):
    st.subheader("Spectrum inspector")

    if "_mz" not in df.columns:
        st.error("This table carries no peak arrays. Re-run `msqc extract` "
                 "with --keep-peaks.")
        return

    c = st.columns([2, 3])
    order = c[0].selectbox("Sort the queue by",
                           ["qc_score (desc)", "longest_tag (desc)",
                            "n_complementary (desc)", "rt_min (asc)",
                            "isolation_purity (asc)"], key="selectbox_604")
    col, asc = {"qc_score (desc)": ("qc_score", False),
                "longest_tag (desc)": ("longest_tag", False),
                "n_complementary (desc)": ("n_complementary", False),
                "rt_min (asc)": ("rt_min", True),
                "isolation_purity (asc)": ("isolation_purity", True)}[order]
    v = view.sort_values(col, ascending=asc).reset_index(drop=True)
    if v.empty:
        st.info("No spectra match the current filters.")
        return

    i = c[1].number_input(f"Spectrum index (0 to {len(v) - 1})", 0,
                          max(len(v) - 1, 0), 0, 1, key="number_input_618")
    row = v.iloc[int(i)]

    m = st.columns(6)
    m[0].metric("Scan", int(row["scan_number"]))
    m[1].metric("m/z", f"{row['precursor_mz']:.4f}")
    m[2].metric("Charge", int(row["charge"]) if pd.notna(row["charge"]) else "?")
    m[3].metric("RT (min)", f"{row['rt_min']:.2f}")
    m[4].metric("Quality", f"{row['qc_score']:.3f}")
    m[5].metric("Bucket", row["triage_class"].replace("_", " "))

    pep_options = {}
    if isinstance(row.get("annotation_error"), str) and row["annotation_error"]:
        st.warning(row["annotation_error"])
    elif isinstance(row.get("rescue_search_modified_peptide"), str) and row["rescue_search_modified_peptide"]:
        pep_options["rescue search"] = row["rescue_search_modified_peptide"]
    if isinstance(row.get("modified_peptide"), str) and row["modified_peptide"]:
        pep_options["search assignment"] = row["modified_peptide"]
    elif isinstance(row.get("peptide"), str) and row["peptide"] and not row.get("annotation_error"):
        pep_options["search assignment"] = row["peptide"]
    if isinstance(row.get("denovo_annotation_error"), str) and row["denovo_annotation_error"]:
        st.warning(row["denovo_annotation_error"])
    elif isinstance(row.get("denovo_peptide"), str) and row["denovo_peptide"]:
        try:
            fragments.parse_peptide(row["denovo_peptide"])
            pep_options["de novo (Casanovo)"] = row["denovo_peptide"]
        except ValueError as exc:
            st.warning(str(exc))
    pep_options["none (raw spectrum)"] = ""
    custom = st.text_input("Or type a peptide to test against this spectrum",
                           placeholder="PEPT[79.9663]IDEK", key="text_input_638")
    if custom:
        pep_options["typed"] = custom

    choice = st.radio("Annotate with", list(pep_options), horizontal=True, key="radio_643")
    pep = pep_options[choice]

    tol = st.slider("Fragment tolerance (Da)", 0.005, 0.5, 0.02, 0.005, key="slider_646")
    st.plotly_chart(
        mirror_plot(row["_mz"], row["_intensity"], pep, row["charge"], tol,
                    title=f"{row['run_id']} scan {int(row['scan_number'])}"
                          + (f"  —  {pep}" if pep else "")),
        use_container_width=True)

    if pep:
        ann = ck.annotate_spectrum(np.asarray(row["_mz"], float),
                                   np.asarray(row["_intensity"], float),
                                   pep, int(row["charge"] or 2),
                                   frag_da=tol)
        k = st.columns(5)
        k[0].metric("Bond coverage", f"{ann['bond_coverage']:.0%}")
        k[1].metric("Longest series", ann["longest_consecutive_series"])
        k[2].metric("Ions matched", ann["n_matched"])
        k[3].metric("Intensity explained", f"{ann['explained_tic_frac']:.0%}")
        e = ann["median_abs_error_ppm"]
        k[4].metric("Median |error|", "n/a" if pd.isna(e) else f"{e:.1f} ppm")
        if ann["bond_coverage"] < 0.5:
            st.warning("Under half the backbone bonds are covered. Ion count "
                       "does not substitute for coverage — ions clustered at "
                       "one terminus localise nothing.")

    with st.expander("All QC features for this spectrum"):
        feats = row.drop(labels=[c for c in ("_mz", "_intensity")
                                 if c in row.index])
        t = pd.DataFrame({"feature": feats.index,
                          "value": feats.astype(str).values})
        t["what it means"] = t["feature"].map(FEATURE_HELP).fillna("")
        st.dataframe(t, use_container_width=True, hide_index=True, height=420)


def page_queue(df, view):
    st.subheader("Rescue queue")
    q = view[view["is_rescue_candidate"]]
    if q.empty:
        st.info("No rescue candidates under the current filters.")
        return

    has_cluster = "cluster_size" in q.columns and q["cluster_size"].notna().any()
    if has_cluster:
        st.caption("Cluster support measures reproducibility. Singletons can be valid; repeated signals still require identification evidence.")
        c = st.columns(3)
        min_size = c[0].number_input("Minimum cluster size", 1,
                               max(int(q["cluster_size"].max()), 1), 1, key="slider_691")
        min_runs = c[1].number_input("Seen in at least N runs", 1,
                               max(int(q["cluster_n_runs"].max()), 1), 1, key="slider_693")
        q = q[(q["cluster_size"].fillna(1) >= min_size)
              & (q["cluster_n_runs"].fillna(1) >= min_runs)]
        c[2].metric("Candidates remaining", f"{len(q):,}")
    else:
        st.info("Open Run rescue to cluster this queue and bring the results back automatically.")

    cols = [c for c in ["run_id", "scan_number", "rt_min", "precursor_mz",
                        "charge", "qc_score", "longest_tag", "n_complementary",
                        "isolation_purity", "cluster_id", "cluster_size",
                        "cluster_n_runs", "denovo_peptide", "denovo_score",
                        "denovo_source", "rescue_status", "rescue_bond_coverage",
                        "rescue_explained_tic_frac", "rescue_precursor_error_ppm",
                        "rescue_search_peptide", "rescue_search_assignment_status"]
            if c in q.columns]
    sort_by = ["cluster_size", "qc_score"] if has_cluster else ["qc_score"]
    show = q[cols].sort_values(sort_by, ascending=False)
    st.dataframe(show, use_container_width=True, hide_index=True, height=430)

    st.download_button("Download this queue as CSV",
                       show.to_csv(index=False).encode(),
                       "rescue_queue.csv", "text/csv", key="download_button_713")

    if "denovo_score" in q.columns and q["denovo_score"].notna().any():
        st.divider()
        st.markdown("**De novo confidence against cluster evidence**")
        st.caption("Top right is where the defensible hits live: confident "
                   "sequence AND reproducible across runs. A single "
                   "high-scoring hit with no cluster support is a hypothesis, "
                   "not a finding.")
        p = q[q["denovo_score"].notna()]
        fig = px.scatter(p, x="denovo_score",
                         y="cluster_size" if has_cluster else "qc_score",
                         size="qc_score", color="qc_score",
                         color_continuous_scale="Blues",
                         hover_data=["run_id", "scan_number", "denovo_peptide"])
        fig.update_layout(height=420, plot_bgcolor="white",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


def page_checklist(df, view):
    st.subheader("PSM interpretation checklist")
    st.caption("Takes the search assignment as given and asks whether it "
               "survives scrutiny. Unlike the rest of this app, it needs a "
               "peptide, so it says nothing about unassigned spectra.")

    c = st.columns(3)
    analyzer = c[0].selectbox(
        "Analyzer", list(ck.ANALYZER_TOLERANCE),
        help="Fragment accuracy is not precursor accuracy. Ion-trap MS2 is "
             "not a ppm instrument. Choosing wrong here will make correct "
             "PSMs look broken.", key="selectbox_742")
    label = c[1].selectbox("Isobaric label", [None, "TMT", "iTRAQ"], key="selectbox_747")
    prec, fppm, fda = ck.ANALYZER_TOLERANCE[analyzer]
    c[2].markdown("**Tolerances**  \nprecursor " + (f"{prec:.0f} ppm" if prec is not None else "not checked") + "  \nfragment "
                  + (f"{fppm:.0f} ppm" if fda is None else f"{fda} Da"))

    if not st.button("Run checklist", type="primary", key="button_752"):
        return
    res = run_checklist_cached(df, analyzer, label)
    if res.empty:
        st.warning("No assigned PSMs with peptides in this table.")
        return

    k = st.columns(4)
    vc = res["verdict"].value_counts()
    k[0].metric("PSMs checked", f"{len(res):,}")
    for j, v in enumerate(["pass", "warn", "fail"]):
        k[j + 1].metric(v, f"{int(vc.get(v, 0)):,}",
                        f"{vc.get(v, 0) / len(res):.1%}")

    reasons = (res.loc[res["concerns"].notna(), "concerns"]
               .str.split("; ").explode()
               .str.replace(r"[\d.]+", "N", regex=True)
               .value_counts().head(10).rename_axis("concern")
               .reset_index(name="n"))
    fig = px.bar(reasons, x="n", y="concern", orientation="h")
    fig.update_layout(height=340, plot_bgcolor="white", yaxis_title="",
                      margin=dict(l=10, r=10, t=10, b=10))
    fig.update_traces(marker_color="#D08A2E")
    st.plotly_chart(fig, use_container_width=True)

    v = st.multiselect("Show verdicts", ["pass", "warn", "fail"],
                       default=["warn", "fail"], key="multiselect_778")
    sub = res[res["verdict"].isin(v)]
    if "one_hit_wonder" in df.columns:
        ohw = df.loc[df["assigned"] & df["one_hit_wonder"].fillna(False),
                     ["run_id", "scan_number"]]
        if len(ohw):
            res = res.merge(ohw.assign(one_hit_wonder=True),
                            on=["run_id", "scan_number"], how="left")
            n = int(res["one_hit_wonder"].fillna(False).sum())
            st.warning(
                f"{n:,} of these PSMs are the only peptide evidence for "
                f"their protein. One-hit wonders are where false PTM "
                f"localisations and spurious protein calls concentrate — "
                f"treat any modification claim on them as unproven.")

    cols = [c for c in ["run_id", "scan_number", "peptide", "verdict",
                        "one_hit_wonder",
                        "bond_coverage", "longest_consecutive_series",
                        "explained_tic_frac", "median_abs_error_ppm",
                        "isolation_purity", "n_top10_unannotated",
                        "delta_alternatives", "concerns"] if c in sub.columns]
    st.dataframe(sub[cols], use_container_width=True, hide_index=True,
                 height=400)
    st.download_button("Download checklist as CSV",
                       res.to_csv(index=False).encode(),
                       "psm_checklist.csv", "text/csv", key="download_button_802")


def page_features(df, view):
    st.subheader("Feature explorer")
    st.caption("Before trusting a model, check what each feature does on its "
               "own. If one feature separates the classes nearly as well as "
               "the model, the model is not earning its complexity.")

    numeric = [c for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c])
               and c not in ("scan_number",)]
    c = st.columns(2)
    feat = c[0].selectbox("Feature", numeric,
                          index=numeric.index("longest_tag")
                          if "longest_tag" in numeric else 0, key="selectbox_817")
    logx = c[1].checkbox("Log X", value=False, key="checkbox_820")
    if feat in FEATURE_HELP:
        st.info(FEATURE_HELP[feat])

    fig = px.histogram(view, x=feat, color="triage_class", barmode="overlay",
                       opacity=0.6, nbins=60,
                       color_discrete_map=BUCKET_COLOURS,
                       log_x=logx, marginal="box")
    fig.update_layout(height=470, plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Single-feature discriminative power**")
    st.caption("AUC for separating assigned from unassigned spectra, one "
               "feature at a time. Values below 0.5 are inverted, not broken — "
               "the feature is anti-correlated.")
    if "assigned" in df and df["assigned"].nunique() == 2:
        from sklearn.metrics import roc_auc_score
        y = df["assigned"].astype(int)
        rows = []
        for cname in numeric:
            x = df[cname]
            if x.notna().sum() < 50 or x.nunique() < 3:
                continue
            try:
                rows.append({"feature": cname,
                             "auc": roc_auc_score(y, x.fillna(x.median()))})
            except Exception:
                pass
        auc = (pd.DataFrame(rows)
               .assign(strength=lambda d: (d["auc"] - 0.5).abs() + 0.5)
               .sort_values("strength", ascending=False).head(18))
        fig = px.bar(auc, x="auc", y="feature", orientation="h")
        fig.add_vline(x=0.5, line_dash="dot", line_color="#999")
        fig.update_traces(marker_color="#5B7FA6")
        fig.update_layout(height=520, plot_bgcolor="white", yaxis_title="",
                          xaxis_title="AUC (assigned vs unassigned)",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.warning("These AUCs use search-engine labels, so they measure "
                   "'looks like an identified tryptic peptide' — not "
                   "'is a good spectrum'. A glycopeptide scores badly here "
                   "and is still worth rescuing. This is exactly why "
                   "`msqc train` holds strong-but-unassigned spectra out of "
                   "training instead of labelling them negative.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def render(tab, fn, *args, name=""):
    """
    Render one tab, containing any failure to that tab.

    Streamlit executes every tab body on each rerun, so an exception raised
    while drawing one tab aborts the script and leaves all later tabs blank.
    That makes a single bug look like five broken features and sends you
    hunting in the wrong place.
    """
    with tab:
        try:
            fn(*args)
        except Exception as e:
            st.error(f"The **{name}** tab failed to render. The other tabs "
                     f"are unaffected.")
            st.exception(e)


def main():
    st.title("msqc")
    st.caption("Spectrum quality triage and dark-proteome rescue")

    tabs = st.tabs(["1 · Load & run", "Overview", "Triage", "Spectrum",
                    "Rescue queue", "PSM checklist", "Features", "Run rescue"])

    render(tabs[0], page_load, name="Load & run")

    qc = st.session_state.get("qc")
    if qc is None:
        for t in tabs[1:]:
            with t:
                st.info("Load data on the first tab, then come back.")
        return

    required = {"run_id", "scan_number", "qc_score", "triage_class"}
    missing = required - set(qc.columns)
    if missing:
        for t in tabs[1:]:
            with t:
                st.error(f"This table is missing {sorted(missing)}. It looks "
                         "like raw qc.parquet — it needs triage applied.")
        return

    view = sidebar_filters(qc)
    st.sidebar.divider()
    st.sidebar.caption(f"{qc['run_id'].nunique()} run(s), {len(qc):,} MS2 "
                       f"spectra loaded")
    if st.sidebar.button("Clear loaded data", key="button_903"):
        for k in ("qc", "qc_key", "qc_source"):
            st.session_state.pop(k, None)
        st.rerun()

    for tab, fn, name in [
            (tabs[1], page_overview, "Overview"),
            (tabs[2], page_triage, "Triage"),
            (tabs[3], page_spectrum, "Spectrum"),
            (tabs[4], page_queue, "Rescue queue"),
            (tabs[5], page_checklist, "PSM checklist"),
            (tabs[6], page_features, "Features"),
            (tabs[7], page_rescue, "Run rescue")]:
        render(tab, fn, qc, view, name=name)


if __name__ == "__main__":
    main()
