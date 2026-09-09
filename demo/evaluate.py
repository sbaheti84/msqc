"""
Check triage output against the simulated ground truth.

Run this after changing thresholds or features. It is the only way to know
whether a change helped, and it is the template for the evaluation you will
need on real data once you have entrapment labels.

    python3 demo/evaluate.py demo_out/qc_triaged.parquet demo_data/*.truth.tsv
"""

from __future__ import annotations

import sys

import pandas as pd

# what triage SHOULD do with each simulated population
EXPECTED = {
    "identified": "identified",
    "unassigned_clean": "rescue_candidate",
    "unassigned_modified": "rescue_candidate",
    "polymer": "polymer_contaminant",
    "noise": "low_quality_unassigned",
}


def main(qc_path, truth_paths):
    qc = pd.read_parquet(qc_path)
    truth = pd.concat([pd.read_csv(p, sep="\t") for p in truth_paths],
                      ignore_index=True)
    keys = ["run_id", "scan_number"] if "run_id" in truth.columns else ["scan_number"]
    df = qc.merge(truth, on=keys, how="inner")
    if df.empty:
        sys.exit("No scans matched between the QC table and the ground truth.")

    print(f"{len(df):,} scans compared\n")

    xt = pd.crosstab(df["true_class"], df["triage_class"])
    print("Rows: what the spectrum actually was. Columns: where triage put it.\n")
    print(xt.to_string())

    df["expected"] = df["true_class"].map(EXPECTED)
    df["correct"] = df["triage_class"] == df["expected"]
    print(f"\nOverall agreement: {df['correct'].mean():.1%}\n")

    per_class = df.groupby("true_class").agg(
        n=("correct", "size"),
        recall=("correct", "mean"),
        mean_qc=("qc_score", "mean"),
    ).round(3)
    print(per_class.to_string())

    # the number that actually matters: of everything sent to the expensive
    # downstream steps, how much was worth sending
    cand = df[df["triage_class"] == "rescue_candidate"]
    if len(cand):
        worth = cand["true_class"].isin(
            ["unassigned_clean", "unassigned_modified"]).mean()
        print(f"\nRescue pile: {len(cand):,} spectra, {worth:.1%} of them are "
              "genuinely interpretable-but-unassigned.")
        contam = cand["true_class"].isin(["polymer", "noise"]).mean()
        print(f"Wasted GPU fraction (noise or polymer in the pile): {contam:.1%}")

    missed = df[(df["true_class"].isin(["unassigned_clean",
                                        "unassigned_modified"]))
                & (df["triage_class"] != "rescue_candidate")]
    if len(missed):
        print(f"\nMissed rescues: {len(missed):,} interpretable unassigned "
              "spectra were not flagged. Where they went:")
        print(missed["triage_class"].value_counts().to_string())
        print(f"Their mean quality score: {missed['qc_score'].mean():.3f} "
              f"(flagged ones: {cand['qc_score'].mean():.3f})")

    if "chimeric" in df.columns:
        print("\nIsolation purity by simulated chimeric status:")
        print(df.groupby("chimeric")["isolation_purity"]
              .agg(["count", "mean", "median"]).round(3).to_string())


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
