# Start here

This is the updated msqc source, based on GitHub commit
`da0a97064d268f24f9d8746ae5fde883636f9a90`.

## Launch

Unzip the package and open a terminal in the `msqc-one-stop` folder.
Use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[app]"
python -m streamlit run app.py
```

On Windows, use `.venv\Scripts\activate` instead of the `source` command.

## Run everything from the dashboard

1. Load your mzML/RAW files and search results in **Load & run**.
2. Run QC, then open **Run rescue**.
3. Select **Built-in** clustering to start without installing another tool.
4. Optionally enable **Casanovo** and **FragPipe**. Enter their installed
   executable paths. Upload or select configuration files as needed.
5. Click **Start rescue workflow**. Watch progress/logs or inspect other tabs.
6. Click **Load rescue results into dashboard** when finished, then inspect
   the queue/spectra or download the ZIP, Parquet, CSV or HTML report.

Falcon, Casanovo, FragPipe and their required dependencies must be installed
on the **Streamlit host**. An executable in another Conda environment can be
selected by its full path. The app runs these tools; it does not replace their
engines or supply GPU hardware. Built-in clustering is available immediately.

For FragPipe, choose a saved workflow and a FASTA. MSBooster and PTM-Shepherd
run when enabled in that workflow. Original mzML files are searched, and the
PSM results are imported automatically. The README explains first-time
headless setup and filtering requirements.

## Main fixes

- Exact run/scan matching and MGF index manifests.
- Explicit PSM confidence filtering and tentative assignments.
- Correct charge-scaled neutral-loss features and backbone cleavage positions.
- Exact modification handling; unsupported or incomplete notation is flagged.
- Working polymer override and neutral `structured_unresolved` label.
- Background jobs with logs, cancellation, timeout, saved stages and downloads.
- Representative prediction mapping without duplicating spectra; cluster
  transfer remains a hypothesis.
- Per-spectrum fragment coverage, explained intensity and precursor error.
- Session-isolated uploads and results retained across dashboard reruns.
- Correct package discovery and mzML-reader dependencies.

## Validation

All 69 tests passed in the development environment. The test suite covers the new runner, Streamlit interactions, identity mapping,
chemistry fixes, external adapter commands, cancellation, timeouts and failures.
A synthetic run processed 1,200 MS2 spectra and completed built-in clustering
for 206 candidates, retaining all input spectrum rows.

External adapters were tested using controlled executable fixtures and checked
against upstream interfaces. Actual Falcon/Casanovo/FragPipe scientific runs
were not performed on experimental data in this environment. Validate your
installed tool versions and scientific settings before production use.

A de novo score or repeated cluster is not an FDR-controlled identification.
The app keeps original assignments and new rescue evidence separate.

## Applying only the changes

The separate `msqc-one-stop.patch` targets the base commit above. From a clean
matching checkout:

```bash
git apply --check /path/to/msqc-one-stop.patch
git apply /path/to/msqc-one-stop.patch
python -m pip install -e ".[app]"
```

These changes have been prepared locally; they have not been pushed to GitHub.
