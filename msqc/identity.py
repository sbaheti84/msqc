"""Lossless MGF export and checked annotation joins."""
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["run_id", "scan_number"]


def attach(qc, annotation):
    """Never enlarge a spectrum table or guess keys during annotation."""
    if qc.duplicated(KEYS).any() or annotation.duplicated(KEYS).any():
        raise ValueError("Duplicate spectrum keys; resolve competing records before joining.")
    if annotation.empty:
        return qc.copy()
    expected = pd.MultiIndex.from_frame(qc[KEYS])
    actual = pd.MultiIndex.from_frame(annotation[KEYS])
    if not actual.isin(expected).all():
        raise ValueError("Annotations contain spectrum IDs absent from the QC table.")
    replace = [c for c in annotation if c in qc and c not in KEYS]
    return qc.drop(columns=replace).merge(annotation, on=KEYS, how="left", validate="one_to_one")


def export_mgf(df, path):
    """MGF uses unique surrogate SCANS; a sidecar preserves original identity.

    Titles retain original run.scan for Falcon. Casanovo index or scan refs
    are resolved through the sidecar, never interpreted as original scans.
    """
    path = Path(path).resolve()
    if df.duplicated(KEYS).any():
        raise ValueError("Duplicate run/scan keys in export.")
    required = {*KEYS, "_mz", "_intensity", "precursor_mz", "charge"}
    if required - set(df):
        raise ValueError(f"Missing export fields: {sorted(required - set(df))}")
    rows = []
    # Validate before opening the output so an invalid row cannot leave a usable-looking MGF.
    for _, row in df.iterrows():
        mz, intensity = np.asarray(row['_mz'], float), np.asarray(row['_intensity'], float)
        if mz.ndim != 1 or mz.size == 0 or mz.shape != intensity.shape:
            raise ValueError("Empty or mismatched peak arrays; re-extract with peaks retained.")
        if not np.isfinite(mz).all() or not np.isfinite(intensity).all() or (mz <= 0).any() or (intensity < 0).any() or intensity.sum() <= 0:
            raise ValueError("Invalid peak values in export.")
        if not np.isfinite(row['precursor_mz']) or row['precursor_mz'] <= 0:
            raise ValueError("Invalid precursor m/z.")
        z = row['charge']
        if not np.isfinite(z) or z < 1 or int(z) != z:
            raise ValueError("Missing or invalid charge; correct it before rescue export.")
        run = str(row['run_id'])
        if any(c in run for c in '\r\n\t'):
            raise ValueError("Run IDs must not contain line breaks or tabs.")
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with path.open('w') as fh:
        for i, row in enumerate(rows):
            title = f"{row['run_id']}.{int(row['scan_number'])}"
            fh.write(f"BEGIN IONS\nTITLE={title}\nSCANS={i+1}\nPEPMASS={row['precursor_mz']:.8f}\nCHARGE={int(row['charge'])}+\n")
            if pd.notna(row.get('rt_min')):
                fh.write(f"RTINSECONDS={row['rt_min']*60:.4f}\n")
            for m, v in zip(row['_mz'], row['_intensity']):
                fh.write(f"{m:.8f} {v:.6f}\n")
            fh.write('END IONS\n')
            manifest.append(dict(mgf_file=path.name, mgf_index=i, export_scan=i+1,
                                 title=title, run_id=row['run_id'], scan_number=int(row['scan_number'])))
    manifest = pd.DataFrame(manifest, columns=['mgf_file','mgf_index','export_scan','title',*KEYS])
    manifest.to_csv(path.with_suffix('.manifest.csv'), index=False)
    return manifest
