"""Small-run greedy clustering; Falcon remains the scalable alternative.

Clusters are anchored to the highest-QC member, without transitive chaining.
This is a triage heuristic, not an identification or FDR procedure.
"""
import math
from collections import defaultdict
import numpy as np
import pandas as pd
from .identity import KEYS


def prepare(row):
    mz, intensity = np.asarray(row['_mz'], float), np.asarray(row['_intensity'], float)
    valid = np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0) & (mz > 0)
    valid &= np.abs(mz - row['precursor_mz']) > 1.5
    mz, intensity = mz[valid], intensity[valid]
    keep = np.argsort(intensity)[-150:]
    mz, intensity = mz[keep], np.sqrt(intensity[keep])
    order = np.argsort(mz)
    norm = np.linalg.norm(intensity)
    return mz[order], intensity[order] / norm if norm else intensity[order]


def similarity(a, b, tolerance):
    """Greedy intensity-product matching with one-to-one peak ownership."""
    am, ai = a; bm, bi = b
    edges = []
    for i, m in enumerate(am):
        lo, hi = np.searchsorted(bm, [m-tolerance, m+tolerance], side='left')
        hi = np.searchsorted(bm, m+tolerance, side='right')
        edges.extend((float(ai[i]*bi[j]), i, j) for j in range(lo, hi))
    ua, ub, value = set(), set(), 0.0
    for weight, i, j in sorted(edges, reverse=True):
        if i not in ua and j not in ub:
            ua.add(i); ub.add(j); value += weight
    return min(value, 1.0), len(ua)


def cluster(df, precursor_ppm=20., fragment_da=.02, min_cosine=.8,
            min_matches=6, progress=None, cancelled=None, max_comparisons=2_000_000):
    if len(df) > 50_000:
        raise ValueError('Built-in clustering supports up to 50,000 spectra. Choose Falcon for larger queues.')
    if not (0 < precursor_ppm <= 500 and fragment_da > 0 and 0 < min_cosine <= 1 and min_matches >= 1):
        raise ValueError('Invalid clustering tolerances.')
    bins, reps, records = defaultdict(list), [], []
    scale = math.log1p(precursor_ppm / 1e6)
    comparisons = 0
    ordered = df.sort_values(['qc_score', *KEYS], ascending=[False, True, True])
    for n, (_, row) in enumerate(ordered.iterrows()):
        if cancelled:
            cancelled()
        mz, z = float(row['precursor_mz']), int(row['charge'])
        b = math.floor(math.log(mz)/scale)
        spec = prepare(row)
        best, value = None, 0.
        # Unknown charge hypotheses must not create confident-looking clusters.
        imputed = bool(row.get('charge_imputed', False))
        candidates = [] if imputed else [k for j in range(b-1, b+2) for k in bins[z, j]]
        for k in candidates:
            rep, rsp = reps[k]
            if abs(mz-rep['precursor_mz']) / rep['precursor_mz'] * 1e6 > precursor_ppm:
                continue
            comparisons += 1
            if comparisons > max_comparisons:
                raise ValueError('Built-in comparison limit reached. Use Falcon for this dense queue.')
            sim, count = similarity(spec, rsp, fragment_da)
            if sim >= min_cosine and count >= min_matches and (best is None or sim > value):
                best, value = k, sim
        if best is None:
            best, value = len(reps), 1.
            reps.append((row, spec))
            if not imputed:
                bins[z, b].append(best)
        rep = reps[best][0]
        records.append(dict(run_id=row['run_id'], scan_number=int(row['scan_number']),
                            cluster_id=f'builtin_{best}', representative_run_id=rep['run_id'],
                            representative_scan_number=int(rep['scan_number']),
                            cluster_similarity=value))
        if progress and n % 100 == 0:
            progress(n/max(len(df), 1), f'Clustered {n:,} / {len(df):,} spectra')
    return summarize(pd.DataFrame(records))


def summarize(members):
    members = members.copy()
    members['cluster_size'] = members.groupby('cluster_id')['scan_number'].transform('size')
    members['cluster_n_runs'] = members.groupby('cluster_id')['run_id'].transform('nunique')
    return members


def from_falcon(df, csv_path):
    from .psm import read_clusters
    parsed = read_clusters(str(csv_path))
    if parsed.duplicated(KEYS).any():
        raise ValueError('Falcon output repeats spectrum IDs.')
    expected = pd.MultiIndex.from_frame(df[KEYS])
    if not pd.MultiIndex.from_frame(parsed[KEYS]).isin(expected).all():
        raise ValueError('Falcon output contains unknown spectrum IDs.')
    # Preserve filtered/unclustered spectra as singletons; never silently lose them.
    members = df[KEYS + ['qc_score']].merge(parsed[KEYS + ['cluster_id']], on=KEYS, how='left', validate='one_to_one')
    members['cluster_id'] = [f'falcon_{c}' if pd.notna(c) else f'singleton_{i}' for i, c in enumerate(members['cluster_id'])]
    reps = members.sort_values('qc_score', ascending=False).drop_duplicates('cluster_id')
    reps = reps.rename(columns={'run_id':'representative_run_id', 'scan_number':'representative_scan_number'})
    members = members.merge(reps[['cluster_id','representative_run_id','representative_scan_number']], on='cluster_id', validate='many_to_one')
    return summarize(members.drop(columns='qc_score'))


def representatives(df, members):
    keys = members[['representative_run_id','representative_scan_number']].drop_duplicates()
    keys.columns = KEYS
    return keys.merge(df, on=KEYS, validate='one_to_one')
