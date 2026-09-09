"""msqc command line interface."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

from . import extract, psm as psm_mod, rescue, report, score

PEAK_COLS = ["_mz", "_intensity"]


def _drop_peaks(df):
    return df.drop(columns=[c for c in PEAK_COLS if c in df.columns])


def _read_qc(path):
    df = pd.read_parquet(path)
    return df


def _extract_one(job):
    """Worker for parallel extraction. Must be module level to be picklable."""
    path, kw = job
    t0 = time.time()
    df = extract.extract_run(path, **kw)
    return path, df, time.time() - t0


def cmd_convert(args):
    from . import convert as cv
    if args.list_backends:
        print("\nRAW conversion backends on this machine:\n")
        for b in cv.available_backends():
            mark = "available" if b["ok"] else "not found"
            print(f"  [{mark:>9}] {b['label']}")
            if not b["ok"]:
                print(f"              install: {b['install']}")
        print(f"\n  Would use: {cv.best_backend() or 'NONE'}\n")
        return
    files = []
    for pat in args.raw:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        sys.exit(f"No files matched: {args.raw}")
    os.makedirs(args.out, exist_ok=True)
    for f in files:
        print(f"[convert] {os.path.basename(f)} …", flush=True)
        try:
            dest = cv.convert_raw(f, args.out, backend=args.backend)
        except Exception as e:
            print(f"[convert]   FAILED: {e}")
            continue
        print(f"[convert]   -> {dest} "
              f"({os.path.getsize(dest) / 1e6:.0f} MB)")


def cmd_check(args):
    """Preflight. Confirms the mzML is usable and that the PSM table will
    actually join to it, before you spend time on a full run."""
    import glob as _glob

    files = []
    for pat in args.mzml:
        files.extend(sorted(_glob.glob(pat)))
    if not files:
        sys.exit(f"No files matched: {args.mzml}")

    print(f"\n=== mzML ({len(files)} file(s)) ===")
    mz_runs, mz_scans, problems = {}, {}, []
    for f in files:
        i = extract.inspect_mzml(f)
        mz_runs[i["run_id"]] = i
        mz_scans[i["run_id"]] = (i["scan_min"], i["scan_max"])
        ratio = i["n_ms2"] / max(i["n_ms1"] + i["n_ms2"], 1)
        print(f"\n  {os.path.basename(f)}")
        print(f"    run_id             {i['run_id']}")
        print(f"    scan id style      {i['id_style']}")
        print(f"    probed             {i['n_ms1']} MS1 / {i['n_ms2']} MS2 "
              f"({ratio:.0%} MS2)")
        print(f"    scan range         {i['scan_min']}-{i['scan_max']}")
        print(f"    RT range           {i['rt_min']:.2f}-{i['rt_max']:.2f} min"
              if i["rt_min"] is not None else "    RT range           unknown")
        print(f"    median MS2 peaks   {i['median_peaks']}")
        print(f"    centroided         {i['centroided']}")

        if i["n_ms2"] == 0:
            problems.append(f"{i['run_id']}: no MS2 scans found at all")
        if i["centroided"] is False:
            problems.append(
                f"{i['run_id']}: PROFILE mode data. Every peak-count and "
                f"noise feature will be meaningless. Re-convert with "
                f"peak picking on (msconvert --filter 'peakPicking vendor "
                f"msLevel=1-', or ThermoRawFileParser default).")
        if i["median_peaks"] and i["median_peaks"] > 2000:
            problems.append(
                f"{i['run_id']}: median {i['median_peaks']:.0f} peaks per MS2 "
                f"suggests profile or unfiltered data.")
        if i["n_ms2"] and i["n_charge_missing"] / i["n_ms2"] > 0.15:
            problems.append(
                f"{i['run_id']}: {i['n_charge_missing'] / i['n_ms2']:.0%} of "
                f"MS2 scans have no charge state. Mass-dependent features "
                f"will guess z=2 on those.")
        if i["n_ms2"] and i["n_no_isolation"] / i["n_ms2"] > 0.1:
            problems.append(
                f"{i['run_id']}: isolation windows missing. isolation_purity "
                f"will be NaN, and that is the most useful feature here.")

    if not args.psm:
        print("\n(no --psm given, skipping the join check)")
    else:
        print(f"\n=== PSM table(s) ===")
        frames = []
        for p in args.psm:
            d = psm_mod.read_psm(p)
            frames.append(d)
            print(f"\n  {os.path.basename(p)}")
            print(f"    engine             {d['engine'].iloc[0] if len(d) else '?'}")
            print(f"    rows               {len(d):,}")
            print(f"    run names          {sorted(d['run_id'].dropna().unique())[:5]}")
            if len(d):
                print(f"    scan range         {int(d['scan_number'].min())}-"
                      f"{int(d['scan_number'].max())}")
                print(f"    decoys             {int(d['is_decoy'].sum()):,}")
        allpsm = pd.concat(frames, ignore_index=True)

        print("\n=== Join check ===")
        psm_runs = set(allpsm["run_id"].dropna().unique())
        mzml_runs = set(mz_runs)
        shared = psm_runs & mzml_runs
        if shared:
            print(f"  run names match: {sorted(shared)}")
            for r in sorted(shared):
                lo, hi = mz_scans[r]
                sub = allpsm[allpsm["run_id"] == r]
                # scan_max comes from a probe of the first few hundred
                # spectra, so it is a lower bound on the true range. Only
                # flag the things a probe can actually prove wrong.
                below = (sub["scan_number"] < lo).mean()
                overlap = sub["scan_number"].between(lo, hi).sum()
                print(f"    {r}: {len(sub):,} PSMs, scans "
                      f"{int(sub['scan_number'].min())}-"
                      f"{int(sub['scan_number'].max())} "
                      f"(mzML probe covered {lo}-{hi})")
                if below > 0.02:
                    problems.append(
                        f"{r}: {below:.0%} of PSM scan numbers are below the "
                        f"mzML's first scan. The psm.tsv came from a "
                        f"different file or a different conversion.")
                elif overlap == 0:
                    print(f"      note: no overlap with the probed window; "
                          f"harmless if the mzML simply has more scans than "
                          f"the probe read.")
        else:
            problems.append(
                "NO RUN NAMES IN COMMON between the mzML and the PSM table.\n"
                f"       mzML says : {sorted(mzml_runs)}\n"
                f"       psm.tsv says: {sorted(psm_runs)[:5]}\n"
                "       Every spectrum would be treated as unassigned and the "
                "rescue pile would be meaningless. Rename the mzML to match "
                "the psm.tsv 'Spectrum' column, or re-run the search on this "
                "exact file.")

    print("\n=== Verdict ===")
    if problems:
        for p in problems:
            print(f"  [!] {p}")
        print("\n  Fix these before running the pipeline.")
        sys.exit(1)
    print("  Everything checks out. Next:\n")
    cmd = f'    msqc run "{args.mzml[0]}"'
    if args.psm:
        cmd += " --psm " + " ".join(args.psm)
    print(cmd + " --out results/")
    print("\n  Then:\n")
    print("    msqc validate --qc results/qc_triaged.parquet "
          "--analyzer orbitrap_hcd")
    print("    streamlit run app.py -- --qc results/qc_triaged.parquet")


def cmd_extract(args):
    files = []
    for pattern in args.mzml:
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        sys.exit(f"No files matched: {args.mzml}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    kw = dict(keep_peaks=args.keep_peaks, max_peaks=args.max_peaks,
              frag_tol=args.frag_tol)
    jobs = [(f, kw) for f in files]

    # One process per file. Extraction is CPU bound and holds only the
    # current MS1 in memory, so this scales linearly until you run out of
    # cores. On GCP Batch you would instead give each file its own task and
    # leave threads at 1.
    threads = max(1, min(getattr(args, "threads", 1), len(files)))
    frames = []
    if threads > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=threads) as ex:
            futures = {ex.submit(_extract_one, j): j[0] for j in jobs}
            for fut in as_completed(futures):
                path, df, elapsed = fut.result()
                _log_extract(path, df, elapsed)
                if not df.empty:
                    frames.append(df)
    else:
        for j in jobs:
            path, df, elapsed = _extract_one(j)
            _log_extract(path, df, elapsed)
            if not df.empty:
                frames.append(df)

    if not frames:
        sys.exit("No MS2 spectra extracted.")
    all_df = pd.concat(frames, ignore_index=True).sort_values(
        ["run_id", "scan_number"]).reset_index(drop=True)
    all_df.to_parquet(args.out, index=False)
    print(f"[extract] wrote {args.out}  ({len(all_df):,} rows, "
          f"{os.path.getsize(args.out) / 1e6:.1f} MB)")

    if getattr(args, "summary", None):
        _write_run_summary(all_df, args.summary)


def _log_extract(path, df, elapsed):
    name = os.path.basename(path)
    if df.empty:
        print(f"[extract] {name}: no MS2 scans found")
        return
    print(f"[extract] {name}: {len(df):,} MS2 scans in {elapsed:.1f}s "
          f"({len(df) / max(elapsed, 1e-6):.0f} scans/s)")


def _write_run_summary(df, path):
    """
    Run-level QC metrics, one row per file. This is the table you watch to
    catch a bad injection or a drifting column before you waste a search on
    it. Field names follow the mzQC vocabulary where one exists.
    """
    import json
    out = []
    for run, g in df.groupby("run_id"):
        rec = {
            "run_id": run,
            "n_ms2": int(len(g)),
            "rt_span_min": float(g["rt_min"].max() - g["rt_min"].min()),
            "median_peaks_above_noise": float(g["n_peaks_above_noise"].median()),
            "median_entropy": float(g["entropy"].median()),
            "median_longest_tag": float(g["longest_tag"].median()),
            "frac_tag_ge3": float((g["longest_tag"] >= 3).mean()),
            "median_isolation_purity": float(g["isolation_purity"].median()),
            "frac_chimeric_purity_lt_0p5": float(
                (g["isolation_purity"] < 0.5).mean()),
            "frac_charge_imputed": float(g["charge_imputed"].mean()),
            "frac_polymer_like": float(g["is_polymer_like"].mean()),
            "median_injection_time_ms": float(g["injection_time_ms"].median()),
            "frac_injection_time_maxed": float(
                (g["injection_time_ms"] >= 0.95 * g["injection_time_ms"].max()).mean())
            if g["injection_time_ms"].notna().any() else None,
        }
        out.append(rec)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[extract] wrote run summary {path}")
    for rec in out:
        flags = []
        if rec["median_isolation_purity"] is not None and \
                rec["median_isolation_purity"] < 0.7:
            flags.append("co-isolation is high")
        if rec["frac_charge_imputed"] > 0.15:
            flags.append("many scans lack a charge state")
        if rec["frac_polymer_like"] > 0.10:
            flags.append("polymer contamination")
        if rec["frac_tag_ge3"] < 0.25:
            flags.append("weak fragmentation overall")
        if flags:
            print(f"[extract]   {rec['run_id']}: " + "; ".join(flags))


def cmd_triage(args):
    df = _read_qc(args.qc)

    if args.psm:
        psms = pd.concat([psm_mod.read_psm(p, args.engine) for p in args.psm],
                         ignore_index=True)
        df = psm_mod.join_psms(df, psms)
    else:
        df["assigned"] = False
        print("  [warn] no PSM table given. Every spectrum will be treated as "
              "unassigned, so the rescue pile will be meaningless.")

    df = score.add_quality_score(df, args.scorer, args.model)
    df = rescue.triage(df, args.qc_threshold, args.min_tag,
                       drop_polymers=not args.keep_polymers)

    print("\n" + rescue.summarise(df).to_string(index=False))

    os.makedirs(args.outdir, exist_ok=True)
    out_parquet = os.path.join(args.outdir, "qc_triaged.parquet")
    df.to_parquet(out_parquet, index=False)
    print(f"\n[triage] wrote {out_parquet}")

    cands = df[df["is_rescue_candidate"]]
    if len(cands) and "_mz" in df.columns:
        mgf = os.path.join(args.outdir, "rescue_candidates.mgf")
        n = rescue.write_mgf(cands, mgf)
        print(f"[triage] wrote {mgf}  ({n:,} spectra)")
        print(rescue.next_step_commands(mgf, args.outdir))
    elif len(cands):
        print("[triage] peak arrays not present; re-run extract with "
              "--keep-peaks to export MGF for falcon/Casanovo.")


def cmd_annotate(args):
    df = _read_qc(args.qc)

    if args.casanovo:
        dn = psm_mod.read_casanovo_mztab(args.casanovo)
        keys = ["scan_number"] if dn["run_id"].eq("").all() else \
            ["run_id", "scan_number"]
        dn = dn[keys + ["denovo_peptide", "denovo_score"]]
        df = df.drop(columns=[c for c in ("denovo_peptide", "denovo_score")
                              if c in df.columns])
        df = df.merge(dn, on=keys, how="left")
        print(f"[annotate] attached {dn['denovo_peptide'].notna().sum():,} "
              "de novo sequences")

    if args.clusters:
        cl = psm_mod.read_clusters(args.clusters)
        # If the cluster file carries no usable run name, fall back to scan
        # number alone. Only safe for a single-run table, so check first.
        keys = ["run_id", "scan_number"]
        if cl["run_id"].eq("").all() or not set(cl["run_id"]) & set(df["run_id"]):
            if df["run_id"].nunique() > 1:
                sys.exit("Cluster file has no run identifiers but the QC table "
                         "spans several runs. Re-run clustering on MGF files "
                         "written by `msqc triage`, which embed run names in "
                         "the TITLE line.")
            cl = cl.drop(columns=["run_id"])
            keys = ["scan_number"]
        df = df.drop(columns=[c for c in ("cluster_id", "cluster_size",
                                          "cluster_n_runs") if c in df.columns])
        df = df.merge(cl, on=keys, how="left")
        n = df["cluster_id"].notna().sum()
        print(f"[annotate] attached cluster assignments for {n:,} spectra")
        if n:
            multi = df.loc[df["cluster_n_runs"].fillna(0) > 1, "cluster_id"].nunique()
            print(f"[annotate]   {df['cluster_id'].nunique():,} clusters, "
                  f"{multi:,} seen in more than one run")

    df.to_parquet(args.out, index=False)
    print(f"[annotate] wrote {args.out}")


def cmd_validate(args):
    df = pd.read_parquet(args.qc)
    if "_mz" not in df.columns:
        sys.exit("This table has no peak arrays. Re-run `msqc extract` with "
                 "--keep-peaks (the `run` command does this by default).")
    out = rescue.validate_psms(df, analyzer=args.analyzer, label=args.label)
    if out.empty:
        sys.exit("No assigned PSMs with peptides found in this table.")
    out.to_parquet(args.out, index=False)

    n = len(out)
    counts = out["verdict"].value_counts()
    print(f"\n[validate] {n:,} assigned PSMs checked "
          f"({args.analyzer} tolerances)\n")
    for k in ("pass", "warn", "fail"):
        c = int(counts.get(k, 0))
        print(f"  {k:5s} {c:6,}  {c / n:6.1%}")

    print("\nMost common concerns:")
    reasons = (out.loc[out["concerns"].notna(), "concerns"]
               .str.split("; ").explode()
               .str.replace(r"[\d.]+", "N", regex=True)
               .value_counts().head(8))
    for reason, c in reasons.items():
        print(f"  {c:6,}  {reason}")

    print(f"\n  median bond coverage           "
          f"{out['bond_coverage'].median():.0%}")
    print(f"  median explained intensity     "
          f"{out['explained_tic_frac'].median():.0%}")
    if out["median_abs_error_ppm"].notna().any():
        print(f"  median fragment error          "
              f"{out['median_abs_error_ppm'].median():.1f} ppm")
    print(f"\n[validate] wrote {args.out}")


def cmd_report(args):
    df = _read_qc(args.qc)
    if "triage_class" not in df.columns:
        sys.exit("Run `msqc triage` before `msqc report`.")
    path = report.write_report(df, args.out, title=args.title,
                               max_spectra=args.max_spectra,
                               peak_cap=args.peak_cap)
    size = os.path.getsize(path) / 1e6
    print(f"[report] wrote {path}  ({size:.1f} MB) — open it in a browser")


def cmd_train(args):
    df = _read_qc(args.qc)
    score.train_model(df, args.out)
    print(f"[train] wrote {args.out}")


def cmd_run(args):
    """extract -> triage -> report in one command."""
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    qc = os.path.join(outdir, "qc.parquet")

    cmd_extract(argparse.Namespace(
        mzml=args.mzml, out=qc, keep_peaks=True,
        max_peaks=args.max_peaks, frag_tol=args.frag_tol,
        threads=getattr(args, "threads", 1),
        summary=os.path.join(outdir, "run_qc.json")))

    cmd_triage(argparse.Namespace(
        qc=qc, psm=args.psm, engine=args.engine, scorer=args.scorer,
        model=args.model, qc_threshold=args.qc_threshold,
        min_tag=args.min_tag, keep_polymers=False, outdir=outdir))

    cmd_report(argparse.Namespace(
        qc=os.path.join(outdir, "qc_triaged.parquet"),
        out=os.path.join(outdir, "report.html"),
        title=args.title, max_spectra=args.max_spectra, peak_cap=120))


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="msqc",
        description="Spectrum-level QC and rescue triage for MS2 data.")
    sub = p.add_subparsers(dest="cmd", required=True)

    cv = sub.add_parser("convert", help="Thermo RAW -> centroided mzML")
    cv.add_argument("raw", nargs="*", help="RAW paths or globs")
    cv.add_argument("--out", default="mzml", help="output directory")
    cv.add_argument("--backend", default=None)
    cv.add_argument("--list-backends", action="store_true",
                    help="show which converters this machine can use")
    cv.set_defaults(func=cmd_convert)

    ch = sub.add_parser("check",
                        help="preflight: is the mzML usable and will the "
                             "PSM table join to it?")
    ch.add_argument("mzml", nargs="+", help="mzML paths or globs")
    ch.add_argument("--psm", nargs="*", default=[],
                    help="FragPipe psm.tsv or Sage results")
    ch.set_defaults(func=cmd_check)

    e = sub.add_parser("extract", help="mzML -> QC feature Parquet")
    e.add_argument("mzml", nargs="+", help="mzML paths or globs")
    e.add_argument("--out", default="qc.parquet")
    e.add_argument("--keep-peaks", action="store_true",
                   help="carry peak arrays through (needed for MGF and report)")
    e.add_argument("--max-peaks", type=int, default=150)
    e.add_argument("--frag-tol", type=float, default=0.02)
    e.add_argument("--threads", type=int, default=1,
                   help="parallel worker processes, one file each "
                        "(leave at 1 on Cloud Batch, where each file is "
                        "already its own task)")
    e.add_argument("--summary", default=None,
                   help="also write run-level QC metrics as JSON")
    e.set_defaults(func=cmd_extract)

    t = sub.add_parser("triage", help="join PSMs, score quality, pick candidates")
    t.add_argument("--qc", required=True)
    t.add_argument("--psm", nargs="*", default=[],
                   help="FragPipe psm.tsv or Sage tsv")
    t.add_argument("--engine", default="auto",
                   choices=["auto", "fragpipe", "sage"])
    t.add_argument("--scorer", default="rule", choices=["rule", "model"])
    t.add_argument("--model", default=None)
    t.add_argument("--qc-threshold", type=float, default=0.6)
    t.add_argument("--min-tag", type=int, default=3)
    t.add_argument("--keep-polymers", action="store_true")
    t.add_argument("--outdir", default="msqc_out")
    t.set_defaults(func=cmd_triage)

    a = sub.add_parser("annotate", help="attach Casanovo and falcon results")
    a.add_argument("--qc", required=True)
    a.add_argument("--casanovo", default=None)
    a.add_argument("--clusters", default=None)
    a.add_argument("--out", default="qc_annotated.parquet")
    a.set_defaults(func=cmd_annotate)

    v = sub.add_parser("validate",
                       help="run the CID/HCD interpretation checklist on "
                            "assigned PSMs")
    v.add_argument("--qc", required=True,
                   help="triaged parquet, extracted with --keep-peaks")
    v.add_argument("--analyzer", default="orbitrap_hcd",
                   choices=["orbitrap_hcd", "orbitrap_cid_it", "tof", "iontrap"],
                   help="sets precursor and fragment mass tolerances. "
                        "Fragment accuracy is NOT the same as precursor "
                        "accuracy and ion-trap MS2 is not a ppm instrument.")
    v.add_argument("--label", default=None, choices=["TMT", "iTRAQ"],
                   help="check for reporter ions")
    v.add_argument("--out", default="psm_checklist.parquet")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("report", help="build the standalone HTML viewer")
    r.add_argument("--qc", required=True)
    r.add_argument("--out", default="report.html")
    r.add_argument("--title", default="Spectrum triage")
    r.add_argument("--max-spectra", type=int, default=1500)
    r.add_argument("--peak-cap", type=int, default=120)
    r.set_defaults(func=cmd_report)

    tr = sub.add_parser("train", help="train a quality model on your own data")
    tr.add_argument("--qc", required=True)
    tr.add_argument("--out", default="msqc_model.pkl")
    tr.set_defaults(func=cmd_train)

    rn = sub.add_parser("run", help="extract + triage + report in one go")
    rn.add_argument("mzml", nargs="+")
    rn.add_argument("--psm", nargs="*", default=[])
    rn.add_argument("--engine", default="auto",
                    choices=["auto", "fragpipe", "sage"])
    rn.add_argument("--outdir", default="msqc_out")
    rn.add_argument("--scorer", default="rule", choices=["rule", "model"])
    rn.add_argument("--model", default=None)
    rn.add_argument("--qc-threshold", type=float, default=0.6)
    rn.add_argument("--min-tag", type=int, default=3)
    rn.add_argument("--threads", type=int, default=1,
                    help="process this many mzML files in parallel")
    rn.add_argument("--max-peaks", type=int, default=150)
    rn.add_argument("--frag-tol", type=float, default=0.02)
    rn.add_argument("--max-spectra", type=int, default=1500)
    rn.add_argument("--title", default="Spectrum triage")
    rn.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
