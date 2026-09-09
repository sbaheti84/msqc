# msqc — spectrum QC and integrated rescue

Spectrum-level quality control and rescue triage for **DDA proteomics**.
Load mzML or Thermo RAW data, check quality, cluster the rescue queue, run
Casanovo or a FragPipe workflow, and inspect the results in one Streamlit app.

## Install and launch

Use Python 3.10 or newer (3.12 recommended for the app):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[app]"
python -m streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate`. The existing
`./run_app.sh` launcher also works after installation.

**Built-in clustering requires no additional tool installation.** It is a
small-run greedy clustering method, not a replacement for Falcon on large
collections. Its limit is 50,000 selected spectra and 2 million spectrum
comparisons; dense datasets can reach the comparison limit earlier.

## Use the app

1. **Load & run**: upload mzML/RAW and optional search tables, or enter paths
   on the Streamlit host. Existing QC Parquet files can also be loaded.
2. Review preflight, select quality/tag settings, and run QC. If PSM q-values
   are missing, explicitly select **Input PSM tables are already FDR-filtered**
   only when that is true. Otherwise their assignments remain tentative.
3. Open **Run rescue**. Select all rescue candidates or the candidates under
   the current sidebar filters. Choose built-in clustering, Falcon, or no
   clustering; optionally enable Casanovo and/or FragPipe.
4. Set tool paths and parameters, then click **Start rescue workflow**.
5. Watch progress and logs while continuing to inspect other tabs. Use
   **Cancel job** to stop an external process. Reruns do not resubmit a job.
6. After completion, click **Load rescue results into dashboard**. Inspect the
   rescue queue and spectrum viewer, or download the complete ZIP, evidence
   CSV, Parquet table, or HTML report.

Every submission gets an isolated directory with an immutable QC snapshot,
settings, MGF identity manifests, logs, intermediate outputs and results.
Failed/cancelled jobs retain completed stages for inspection; retry by starting
a new job. Job history is available during the browser session. Files remain
on disk after the session ends: load `result.parquet` or `partial.parquet`
through **Load & run**. Set `MSQC_WORKDIR` to a persistent directory before
launching if the host clears its temporary directory.

### External tools: configure once, then run from Streamlit

Tools run **on the machine hosting Streamlit**, not on the browser user's
computer. The executable fields accept a command on PATH or the full path to
an executable in a separate environment. This avoids requiring the app and
all machine-learning dependencies to share one environment.

| Tool | Setup | App behavior |
|---|---|---|
| Built-in clustering | Included | Matches precursor charge/mass and fragments to a fixed highest-QC representative; retains singletons |
| Falcon | Install `falcon-ms` in a compatible environment | Detects supported distance option, runs clustering, retains filtered/unclustered spectra as singletons |
| Casanovo | Install `casanovo` in a compatible environment; configure its model and CPU/GPU settings | Sequences representatives, imports mzTab, maps predictions to original scans and cluster members |
| FragPipe | Install/configure FragPipe and the tools required by the selected workflow | Runs headless on original mzML files and imports `psm.tsv` |

For example, in an appropriate dedicated environment:

```bash
python -m pip install falcon-ms
# In the Casanovo environment:
python -m pip install casanovo
```

The app checks executable availability and displays a useful failure if a
required tool is missing. It does not silently install large dependencies.
Use a Linux workstation/server with sufficient memory and, for faster de novo
sequencing, a compatible GPU. A resource-limited Streamlit hosting service may
support the dashboard and built-in clustering but not large external jobs.

Casanovo's optional YAML file controls device selection, batch size and model
settings. Without a supplied checkpoint it may download compatible weights.
The adapter supports the current `--output_dir` / `--output_root` interface
and older versions exposing `--output`; unrecognized interfaces fail visibly.

### FragPipe searches, MSBooster and PTM-Shepherd

Upload or supply paths to a saved `.workflow`, a FASTA with the decoys expected by that workflow,
and the **original mzML files**. Use an Open, mass-offset, semi-tryptic, or other
workflow configured for your experiment. The app inserts the selected FASTA
into a copy of the workflow and leaves scientific search settings unchanged.

MSBooster and PTM-Shepherd run when enabled in that workflow. Their original
outputs are retained in the job's `fragpipe/` directory and download bundle.
They are not independently reimplemented by msqc.

For first-time headless setup, provide the FragPipe tools directory, DIA-NN
executable and Python directory as needed. Enable PSM reporting. Only declare
that the workflow returns FDR-filtered PSMs when its filtering has been
configured accordingly. A user-entered maximum q-value filters reported
q-values; it does **not** rewrite FragPipe's FDR settings or calculate a new FDR.

FragPipe searches the original files rather than a pooled rescue MGF. This
preserves native identity and acquisition context and avoids treating a
selected, pooled subset as an ordinary full search. Results are attached using
exact `(run_id, scan_number)` keys.

## What the output means

| Field | Meaning |
|---|---|
| `assigned`, `triage_class` | Original QC/search baseline, preserved during rescue |
| `assignment_status` | `confident`, `tentative`, or `unassigned` for initial imported PSMs |
| `cluster_id`, `cluster_size`, `cluster_n_runs` | Cluster membership and reproducibility context |
| `representative_run_id`, `representative_scan_number` | Original identity of the highest-QC representative |
| `denovo_peptide`, `denovo_score` | Top prediction and the tool's raw score |
| `denovo_candidate_count`, `denovo_score_gap` | Number of reported alternatives and top/runner-up gap |
| `denovo_source` | `direct_prediction` or `cluster_hypothesis`; cluster transfer is not identification acceptance |
| `rescue_bond_coverage`, `rescue_explained_tic_frac` | Sequence-specific fragment evidence on each member spectrum |
| `rescue_precursor_error_ppm` | Observed versus sequence-calculated precursor m/z error |
| `rescue_search_*` | Imported FragPipe search fields, separate from the original baseline |
| `rescue_status` | `unresolved`, `sequence_hypothesis`, or `search_confident` |

**A quality score, high de novo score, or large cluster is not a peptide
identification confidence estimate.** Singletons can be valid; repeatable
contaminants can form large clusters. Class-specific and rescue-subset error
assessment remain necessary for scientific claims. No new target-decoy or
entrapment FDR estimator is implemented here.

The five QC buckets are `identified`, `rescue_candidate`,
`structured_unresolved`, `polymer_contaminant`, and `low_quality_unassigned`.
`structured_unresolved` replaces the overly definitive
`structured_non_peptide` label. Lack of a sequence tag does not prove that a
spectrum is non-peptide. Thresholds are configurable heuristics, not calibrated
probabilities.

## Chemistry and identity safeguards

- Native run/scan identity must match. There is no automatic scan-only fallback
  across mismatched runs. Programmatic callers can supply `run_mapping` to
  `join_psms` for an explicitly verified rename.
- MGF exports include `.manifest.csv` mapping zero-based MGF indices and
  surrogate export scans to original spectra. mzTab indices are never treated
  as original scan numbers. Missing prediction rows do not shift later matches.
- Competing de novo predictions are ranked without duplicating QC rows; their
  full raw mzTab remains available. Cluster-member predictions are hypotheses.
- Neutral-loss m/z features account for precursor charge. b/y complementary
  cleavages use the same backbone coordinates in the checklist.
- FragPipe's rounded total-residue masses are not treated as delta masses.
  Exact Assigned Modifications are used where available, with peptide-mass
  consistency checking when the table supplies a calculated mass. Incomplete
  fixed/terminal modification information is flagged rather than guessed.
- Fragment annotation supports numeric delta notation, including numeric
  ProForma terminal deltas. Unsupported named/ambiguous modifications are
  flagged; they are not silently stripped. Such records remain visible.
- Missing mzML charges are still imputed as 2+ by extraction and flagged.
  Built-in clustering keeps these spectra as singletons. Automatic precursor
  charge/isotope correction is not implemented.

## RAW conversion and preflight

RAW conversion still requires ThermoRawFileParser or the supported Docker
backend. Use **the exact conversion searched by your search engine**.

```bash
msqc convert --list-backends
msqc check "data/*.mzML" --psm fragpipe/psm.tsv
```

Profile-mode data is unsuitable for these peak features. Incomplete mzML files
are flagged and parsable spectra can be salvaged, but per-run rates then refer
to only part of the gradient. Prefer host paths over multi-GB browser uploads.

## CLI

```bash
msqc run "data/*.mzML" --psm fragpipe/psm.tsv --assume-prefiltered --outdir results
msqc extract "data/*.mzML" --keep-peaks --out qc.parquet
msqc triage --qc qc.parquet --psm fragpipe/psm.tsv --assume-prefiltered --outdir results
msqc annotate --qc results/qc_triaged.parquet --casanovo predictions.mztab \
  --manifest results/rescue_candidates.manifest.csv --out results/annotated.parquet
msqc report --qc results/annotated.parquet --out results/report.html
```

Use the manifest for the **exact MGF supplied to Casanovo**. Manifests for a
full queue cannot resolve an unrelated representative MGF. The Streamlit
runner generates and uses the correct manifest automatically.

## Validation and limits

```bash
python -m pip install pytest
python -m pytest tests -q
```

Tests cover chemistry, confidence-aware joins, MGF/mzTab identity, clustering,
background execution, failures, cancellation, external adapter contracts and
Streamlit interactions. Controlled executable fixtures test external-tool
orchestration; they do not replace running your actual installed tools on
experimental data. The synthetic demo tests plumbing, not biological accuracy.

`msqc train` requires at least two runs. Single-feature baselines use the same
holdout as the model. De novo scores alone are not positive labels. Run splitting
still permits recurring peptides across runs; external and peptide-disjoint
validation are needed for stronger generalization claims.

The Streamlit table remains memory-resident; the built-in method and HTML
viewer are bounded for moderate datasets. DIA, automatic charge/isotope
correction, new FDR estimation and full arbitrary ProForma support remain out
of scope. External jobs continue through normal Streamlit reruns; host shutdown
still interrupts computation. Only expose this local-file/process-running app
to trusted users on an appropriately secured host.

### Upstream interfaces

- [Falcon source and CLI](https://github.com/bittremieux/falcon)
- [Casanovo CLI setup](https://casanovo.readthedocs.io/en/latest/getting_started.html)
- [FragPipe headless execution](https://fragpipe.nesvilab.org/docs/tutorial_headless.html)
- [FragPipe result formats](https://fragpipe.nesvilab.org/docs/tutorial_fragpipe_outputs.html)
