# msqc

Spectrum-level QC and dark-proteome rescue triage for DDA proteomics.

Scores every MS2 spectrum from the mzML alone — no FASTA, no search engine —
then sorts spectra into buckets so you know which unassigned scans are worth
sending to clustering, de novo sequencing, or an open search, and which are
noise or detergent.

---

## What this is for, and what it is not for

**It is for triage.** Spectrum quality scoring decides where to spend GPU
time. On its own it does not find biology.

**It is not a discovery tool.** If your goal is more identifications, do these
first, in this order, because each one yields more per hour of work than
anything in this repo:

1. Rescoring — MSBooster inside FragPipe, or MS2Rescore. Usually 10–30% more
   PSMs at the same FDR, for configuration rather than code.
2. Precursor mass and charge correction, plus deisotoping, in your search
   parameters. A large share of "unassigned" spectra are fine spectra with the
   wrong monoisotopic peak selected.
3. Open or mass-offset search, then PTM-Shepherd.
4. Semi-tryptic search.

Run `msqc` after those, on what is left.

**On novel/cryptic peptides.** Expanding a FASTA with lncRNA or UTR ORFs
inflates the search space enormously while the number of true novel peptides
stays tiny. A global 1% FDR then hides a novel-class FDR that can be far
higher. Feeding de novo output back into a search is a two-stage procedure
that breaks target-decoy assumptions outright. If you intend to publish
anything from the rescue pile, plan for class-specific FDR, entrapment
databases, and synthetic peptide validation. This tool will happily hand you a
queue of interesting-looking spectra; it will not make them true.

---

## Install

```bash
pip install -e .
pip install -e ".[ml]"   # optional: LightGBM for the trained scorer
```

Requires Python ≥3.10, numpy, pandas, pyarrow, pyteomics. scikit-learn is used
for training; LightGBM is used if present and sklearn's histogram gradient
booster is the fallback.

## Thermo RAW files

You can hand the app or the CLI `.raw` files directly. Nothing in Python reads
Thermo RAW, so msqc shells out to a converter. Check what your machine can do:

```bash
msqc convert --list-backends
```

Install one (in order of preference):

```bash
conda install -c bioconda thermorawfileparser   # simplest, no Docker
# or just start Docker Desktop — msqc uses a pinned container
```

Then either let the pipeline convert on the fly, or do it up front:

```bash
msqc convert "raw/*.raw" --out mzml/
```

Conversion always produces **centroided, indexed** mzML. That is not
configurable: profile-mode data makes every peak-count, noise and
signal-to-noise feature in this package meaningless, so peak picking is never
optional. Output is cached and reused when the mzML is newer than the RAW,
because conversion is the slowest step in the whole pipeline.

### The trap to avoid

**Your psm.tsv must come from the same conversion.** If FragPipe searched an
mzML converted with different settings, scan numbers may not line up and
nothing will join. Preflight compares run names, but it cannot detect two
conversions of the same run that number scans differently.

The safe order is: convert once, search that mzML, then run msqc on the same
file. If you already have FragPipe results, feed msqc the exact mzML that
FragPipe searched rather than re-converting the RAW.

## Step 0: preflight

Always run this first on real data. It catches the failure that produces a
silent 0% identification rate.

```bash
msqc check "data/*.mzML" --psm fragpipe/psm.tsv
```

Reports whether the mzML is centroided, how many MS2 scans it has, whether
charge states and isolation windows are present, and — most importantly —
**whether the run names in psm.tsv match the mzML filename**. FragPipe writes
`Spectrum` as `<run>.<scan>.<scan>.<charge>`, and msqc derives `run_id` from
the mzML filename. If your mzML has been renamed since the search, nothing
joins, every spectrum is treated as unassigned, and the rescue pile is
garbage. `check` exits non-zero rather than letting that through.

## Quickstart

```bash
# everything in one command
msqc run "data/*.mzML" --psm fragpipe/psm.tsv --out results/
```

Or step by step:

```bash
# 1. mzML -> QC features. No database involved.
msqc extract "data/*.mzML" --out qc.parquet --keep-peaks \
     --threads 8 --summary run_qc.json

# 2. join your search results, score quality, bucket the spectra
msqc triage --qc qc.parquet --psm fragpipe/psm.tsv --outdir results/

# 3. cluster the rescue pile, then de novo only the representatives
falcon results/rescue_candidates.mgf results/falcon \
       --export_representatives --precursor_tol 20 ppm \
       --fragment_tol 0.05 --eps 0.1
casanovo sequence -o results/casanovo.mztab results/falcon.mgf

# 4. bring the results back in
msqc annotate --qc results/qc_triaged.parquet \
     --casanovo results/casanovo.mztab \
     --clusters results/falcon.csv \
     --out results/qc_annotated.parquet

# 5. standalone HTML viewer, no server
msqc report --qc results/qc_annotated.parquet --out results/report.html
```

`msqc triage` prints the exact commands for step 3 with your paths filled in.
Those tools are deliberately not wrapped — their parameters are where the FDR
risk lives, and hiding them behind a wrapper would be a disservice.

---

## The triage buckets

Five mutually exclusive classes:

| Bucket | Meaning | What to do |
|---|---|---|
| `identified` | Search engine explained it | Nothing |
| `rescue_candidate` | Strong peptide-like structure, unassigned, not a polymer | Cluster → open search → de novo |
| `structured_non_peptide` | Good signal, no peptide fragmentation pattern | Glycans, lipids, crosslinks. Different tools |
| `polymer_contaminant` | Evenly spaced repeat ladder | Fix your sample prep |
| `low_quality_unassigned` | Not enough signal to interpret | Nothing. This is most of the pile |

The polymer bucket earns its place: a PEG ladder scores well on every
peptide-agnostic quality metric and will otherwise fill your GPU queue with
detergent.

---

## Features extracted

All computed from the peak list and the preceding MS1 survey scan.

**Acquisition context** — retention time, precursor m/z and charge, whether
charge had to be imputed, injection time, precursor intensity,
**isolation purity**, number of co-fragmented species.

Isolation purity is the single best chimeric-interference indicator and the
one most pipelines forget. Below roughly 0.5 the MS2 is a mixture and no
search engine will explain it with one peptide.

**Signal quality** — peaks above an estimated noise floor, TIC, base peak,
signal-to-noise proxy, spectral entropy, normalised entropy, top-10 and top-20
intensity fraction, log dynamic range.

**Peptide-like structure** — residue-mass gap count and density, **longest
sequence tag**, tag-length histogram, **b/y complementary pairs** and their
intensity fraction, isotope cluster count and intensity fraction, fraction of
intensity above the precursor, water/ammonia/phospho neutral losses.

**Contaminant signatures** — longest evenly spaced ladder, its repeat mass and
label (PEG, PPG, siloxane, alkyl), polymer flag.

Structural features are peptide-centric by construction. They are kept
separate from signal-quality features so you can find spectra that are
well-structured but *not* peptide-like, rather than discarding them.

---

## Scoring

**`--scorer rule`** (default) — a transparent weighted composite. No training
data needed, works on the first file you ever process. Ship this first.

**`--scorer model`** — gradient-boosted classifier trained on your own data:

```bash
msqc train --qc results/qc_triaged.parquet --out msqc_model.pkl
```

Two things this does that most quality models do not:

**Three label classes, not two.** Spectra that are structurally strong but
unassigned are labelled `-1` and **held out of training**. Every published
spectrum-quality model takes its labels from a database search, so it learns
"does this look like a confidently identified tryptic peptide". Trained
naively, it would score down exactly the glycopeptides, crosslinks, and
non-tryptic cryptic peptides you are trying to rescue. There is a test that
fails if this guard is removed.

**Single-feature AUC baselines are printed.** If the model beats the best
single feature by less than 0.03 AUC, it says so and tells you to stay with
`--scorer rule`. Splits are by run, never random — a random split leaks the
same peptide across both sides and reports a fake AUC near 0.99.

---

## Evaluating changes

```bash
python3 demo/make_demo_data.py --out demo_data
msqc run "demo_data/*.mzML" --psm demo_data/*.psm.tsv --out demo_out
python3 demo/evaluate.py demo_out/qc_triaged.parquet demo_data/*.truth.tsv
python3 -m pytest tests -q
```

On the bundled synthetic data: 90.6% recall and 100% precision on rescuable
spectra, 80/80 polymers correctly diverted.

**Do not read anything into that.** The synthetic noise spectra are trivially
separable from peptide spectra; the AUC is 1.000 because the problem is fake.
Real data is a continuum. What the demo proves is that the plumbing is
correct — keys join, buckets are exclusive, no rows are lost or duplicated. It
proves nothing about generalisation. Your first honest number comes from
`msqc train` on your own runs, reading the single-feature baselines.

---

## Running on GCP

Extraction is embarrassingly parallel per file and holds only the current MS1
in memory.

```
GCS (Thermo RAW)
  → ThermoRawFileParser container → mzML in GCS
  → Cloud Batch, spot VMs, one task per file:
        msqc extract $FILE --out gs://.../qc/$RUN.parquet --keep-peaks --threads 1
  → search (FragPipe, or Sage for cheap parameter sweeps)
  → msqc triage  → rescue_candidates.mgf
  → falcon across all runs  (CPU, collapses the GPU workload)
  → Casanovo on cluster representatives only  (short-lived GPU pool, spot)
  → msqc annotate → msqc report
```

Leave `--threads 1` on Cloud Batch, where each file already has its own task.
Use `--threads N` only for local multi-file runs.

Cluster before de novo, always. It cuts GPU spend by roughly an order of
magnitude and, more importantly, `cluster_size` and `cluster_n_runs` are your
strongest evidence filter. A novel peptide seen once is noise. Seen in 40
spectra across 12 samples, it is a real molecule.

---

## The Streamlit app

The app runs the whole pipeline itself. You do not need to touch the CLI.

```bash
pip install -e ".[app]"
./run_app.sh
```

**Tab 1, Load & run.** Three ways to get data in:

- **Upload files** — drag in your mzML files, `psm.tsv`, and optionally
  `peptide.tsv` and `protein.tsv`. Uploads stream to disk rather than sitting
  in memory. `.streamlit/config.toml` raises the upload cap to 4 GB, but
  browser uploads of multi-GB mzML are slow; use paths for those.
- **Paths on this machine** — text boxes taking paths or globs
  (`/data/hela/*.mzML`). The right option for anything large or for a shared
  server.
- **Existing qc_triaged.parquet** — reload earlier CLI output.

It then runs a **preflight** and shows a per-file table: MS1/MS2 counts,
centroided or profile, median peaks, scan range, missing charge states,
missing isolation windows. Blocking problems (profile-mode data, no MS2, and
above all **no run names in common between the mzML and psm.tsv**) stop the
run rather than producing a plausible-looking wrong answer.

Set the scorer, quality threshold, minimum tag and peak cap, press
**Run QC pipeline**, and watch a real progress bar. The result stays in
session state for every other tab and can be downloaded as
`qc_triaged.parquet` to continue on the command line.

### What peptide.tsv and protein.tsv are for

They are optional and used for context only. `protein.tsv` gives peptide
counts per protein, which yields a **one-hit-wonder flag**: a PSM that is the
only peptide evidence for its protein. One-hit wonders are where false PTM
localisations and spurious protein calls concentrate, so the PSM checklist tab
warns when your flagged PSMs are concentrated among them.

The app and the CLI call the same `msqc.pipeline.run_pipeline`, so the
dashboard cannot drift from the batch job.

### The remaining six tabs, with sidebar filters (run, triage class, quality range, charge,
retention time, minimum tag, minimum isolation purity) applied across all of
them.

**Overview** — headline metrics with the three warnings that matter (low
isolation purity, missing charge states, polymer load), bucket breakdown, and
identification rate across the gradient overlaid with median isolation purity.
When those two collapse together it is a co-isolation problem, not a
chromatography problem.

**Triage** — interactive scatter with selectable axes, coloured by bucket,
hover showing scan and peptide. Below it a **threshold explorer**: move the
quality and tag cut-offs and watch the queue size respond before committing
GPU time.

**Spectrum** — the main working view. Pick a spectrum, annotate it with the
search assignment, the Casanovo call, or **a peptide you type in**, and get a
mirror plot with b/y ions labelled. Live readout of bond coverage, longest
consecutive series, intensity explained, and median ppm error. Adjustable
fragment tolerance. Full feature table with plain-language explanations.

**Rescue queue** — filter by cluster size and how many runs a cluster spans,
sort, export to CSV. Plots de novo confidence against cluster evidence: the
top-right corner is where defensible hits live.

**PSM checklist** — runs the interpretation checklist with a selectable
analyzer preset, shows pass/warn/fail, ranks the most common concerns, filters
to the problem PSMs.

**Features** — distribution of any feature split by bucket, plus single-feature
AUCs. This tab is deliberately blunt about the fact that those AUCs use
search-engine labels and therefore measure "looks like an identified tryptic
peptide", not "is a good spectrum".

### Honest scale limit

Streamlit reruns the whole script on every widget interaction. Everything
expensive is behind `st.cache_data` keyed on file path and mtime, and the
scatter samples above a configurable point count. Comfortable to roughly 200k
spectra on a laptop. Past that the layout is fine but you will want the
filtering pushed into DuckDB instead of a dataframe — the tabs would not
change, only `load_qc`.

## The standalone HTML viewer

`report.html` is a single self-contained file. No server, no Streamlit, no
build step. Open it, or drop it in a GCS bucket and share the link.

Contains a triage scatter (quality vs search score, coloured by bucket), a
sortable rescue queue, a canvas spectrum viewer, and a per-spectrum feature
readout.

It embeds spectra as JSON, which is comfortable to roughly 20k spectra. Past
that, switch to DuckDB-WASM reading the Parquet over HTTP range requests —
same data model, no code here needs to change.

---

---

## `msqc validate` — automated PSM interpretation checklist

Separate from the QC pipeline. The rest of msqc scores spectra *without* an
assignment; this takes an assignment as given and asks whether it survives a
manual-interpretation checklist.

```bash
msqc validate --qc results/qc_triaged.parquet --analyzer orbitrap_hcd
msqc validate --qc results/qc_triaged.parquet --analyzer orbitrap_cid_it --label TMT
```

Checks per PSM: backbone bond coverage and longest consecutive ion series;
fraction of intensity the assignment explains; how many of the top ten peaks
are unexplained, cross-referenced against isolation purity; fragment and
precursor mass error against **analyzer-specific** tolerances; PTM diagnostic
ions; delta-mass alternative explanations; immonium consistency; cleavage
chemistry (proline effect, basic-residue charge sequestration, missed
cleavages); TMT/iTRAQ reporter ions.

### Fragment tolerance is not precursor tolerance

`--analyzer` exists because a single mass-error rule is wrong. Presets:

| Analyzer | Precursor | Fragment |
|---|---|---|
| `orbitrap_hcd` | 5 ppm | 20 ppm |
| `orbitrap_cid_it` | 5 ppm | 0.4 Da (ion trap MS2 is not a ppm instrument) |
| `tof` | 10 ppm | 25 ppm |
| `iontrap` | — | 0.5 Da |

Holding Orbitrap HCD *fragments* to 2 ppm would reject nearly every correct
PSM. Low-intensity and low-m/z fragments routinely land at 5-20 ppm.

### Verdicts are reasons, not a score

`pass` / `warn` / `fail` plus a list of concerns. A PSM failing one hard check
is not the same as one tripping three soft ones, and a single number hides
which. Co-isolation is a `warn`, never a `fail` — it explains unassigned
peaks, it does not make the identification wrong.

### Reference tables

`checklist.py` encodes PTM signature ions (phospho-Y, acetyl-K, mono/di/tri
methyl on K and R, ADP-ribosyl) and delta-mass ambiguities. The ambiguity
table flags the pairs that need resolving power to separate, and reports how
much:

- acetyl 42.010565 vs trimethyl 42.046950 — 0.036 Da, needs R ≈ 27,000 at
  m/z 1000, so judge it on the MS1 precursor and never on a low-resolution MS2
- formyl 27.994915 vs dimethyl 28.031300 — the same trap, rarely mentioned
- **deamidation 0.984016 vs a 13C isotope error 0.997035** — the most common
  false PTM of all, and the one absent from most checklists

## Truncated mzML files

`XMLSyntaxError: Premature end of data` deep inside a file means the mzML is
incomplete, not that the reader is broken. Common causes: an interrupted
conversion, a killed download or rsync, a browser upload that stopped short,
or a job that hit a disk quota.

msqc handles this in three places:

1. **Preflight seeks to the end of the file** and checks for a closing
   `</mzML>` tag before parsing anything. One seek, so it always runs.
2. **Extraction salvages** whatever parsed before the tear rather than
   discarding the run. A file torn at 90% still holds 90% of a usable run.
3. **The result is labelled INCOMPLETE**, because every per-run rate is then
   computed over a partial gradient — and the missing part is almost always
   the end of the gradient, which is not a random sample of your peptides.

To confirm by hand:

```bash
tail -c 200 yourfile.mzML     # a complete file ends </mzML></indexedmzML>
```

If it was uploaded through the browser, re-transfer it and use the
**Paths on this machine** option instead. Multi-GB uploads through a browser
are the most common source of this error.

## Known limitations

- **`qc_threshold=0.6` and `min_tag=3` are guesses.** Tune them against your
  own identification rate before trusting the queue size.
- **Noise floor is a percentile estimate.** Fine for centroided Orbitrap data,
  probably wrong where a hard intensity threshold was applied at acquisition.
- **Isotope detection is a cheap pairwise check**, not real deisotoping. Use
  pyOpenMS or mzdata if you need the genuine article.
- **Single-charge-state assumption** when charge is missing from the mzML.
  Watch `frac_charge_imputed` in `run_qc.json`.
- **No entrapment FDR built in.** You have to add distant-species proteins to
  your FASTA yourself. Do it before you believe any rescued peptide.
- **DIA is out of scope.** This assumes DDA with one dominant precursor per
  MS2.

## If you move the hot loop to Rust

The residue-gap adjacency matrix and the ladder walk are the only real
bottlenecks. `mzdata` reads Thermo RAW natively, which would let you delete
the mzML conversion stage entirely, and `rustyms` covers fragment generation
and annotation. Wrap with PyO3/maturin and keep the orchestration in Python —
do not rewrite the pipeline.

## Layout

```
msqc/features.py   per-spectrum metrics (no database)
msqc/extract.py    mzML streaming, parallel across files
msqc/psm.py        FragPipe / Sage / Casanovo mzTab / falcon / MaRaCluster readers
msqc/score.py      rule scorer, label construction, model training
msqc/rescue.py     triage buckets, MGF export, next-step commands
msqc/fragments.py  theoretical b/y ions for the viewer
msqc/report.py     standalone HTML builder
msqc/checklist.py  PTM signature ions, delta-mass ambiguities, PSM verdicts
msqc/cli.py        extract / triage / annotate / validate / report / train / run
app.py             Streamlit dashboard (six tabs)
run_app.sh         launcher
tests/             20 regression tests
demo/              synthetic data generator and ground-truth evaluator
```
