"""
Spectrum-level QC feature extraction.

Everything here operates on plain numpy arrays so it can be swapped for a
Rust/PyO3 implementation later without changing callers. The functions that
dominate runtime are marked HOTLOOP.
"""

from __future__ import annotations

import numpy as np

PROTON = 1.00727646688
NEUTRON = 1.00335483507

# Residue monoisotopic masses. I and L are identical; K and Q differ by 0.036
# which is resolvable at high res, so both are kept.
RESIDUE_MASSES = {
    "G": 57.02146, "A": 71.03711, "S": 87.03203, "P": 97.05276,
    "V": 99.06841, "T": 101.04768, "C": 103.00919, "L": 113.08406,
    "I": 113.08406, "N": 114.04293, "D": 115.02694, "Q": 128.05858,
    "K": 128.09496, "E": 129.04259, "M": 131.04049, "H": 137.05891,
    "F": 147.06841, "R": 156.10111, "Y": 163.06333, "W": 186.07931,
}
RESIDUES = np.array(sorted(set(RESIDUE_MASSES.values())))

# Repeating mass differences that indicate a polymer rather than a peptide.
POLYMER_DELTAS = {
    44.0262: "PEG",
    58.0055: "polypropylene glycol / PEG variant",
    74.0186: "polysiloxane-related",
    88.0524: "Triton-like",
    162.0528: "hexose ladder",
}

H2O = 18.010565
NH3 = 17.026549


# --------------------------------------------------------------------------
# peak list preparation
# --------------------------------------------------------------------------

def top_n(mz: np.ndarray, inten: np.ndarray, n: int = 150):
    """Keep the n most intense peaks, returned sorted by m/z."""
    if mz.size <= n:
        order = np.argsort(mz)
        return mz[order], inten[order]
    idx = np.argpartition(inten, -n)[-n:]
    idx = idx[np.argsort(mz[idx])]
    return mz[idx], inten[idx]


def estimate_noise(inten: np.ndarray) -> float:
    """Robust noise floor. Median of the lower half of the intensity list."""
    if inten.size == 0:
        return 0.0
    lower = np.sort(inten)[: max(1, inten.size // 2)]
    return float(np.median(lower))


# --------------------------------------------------------------------------
# signal-quality features
# --------------------------------------------------------------------------

def spectral_entropy(inten: np.ndarray) -> float:
    total = inten.sum()
    if total <= 0:
        return 0.0
    p = inten / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def normalised_entropy(inten: np.ndarray) -> float:
    """Entropy divided by log(n). 1.0 = flat noise, low = a few dominant peaks."""
    if inten.size < 2:
        return 0.0
    return spectral_entropy(inten) / np.log(inten.size)


# --------------------------------------------------------------------------
# structural features  (HOTLOOP)
# --------------------------------------------------------------------------

def gap_adjacency(mz: np.ndarray, tol: float = 0.02) -> np.ndarray:
    """
    Upper-triangular boolean matrix: adj[i, j] is True when mz[j] - mz[i]
    matches a residue mass. Peaks must be sorted by m/z.
    """
    n = mz.size
    if n < 2:
        return np.zeros((n, n), dtype=bool)
    diff = mz[None, :] - mz[:, None]
    hit = (np.abs(diff[:, :, None] - RESIDUES[None, None, :]) < tol).any(axis=2)
    return np.triu(hit, k=1)


def longest_tag(adj: np.ndarray) -> int:
    """Longest chain of consecutive residue-mass gaps (a sequence tag)."""
    n = adj.shape[0]
    if n == 0:
        return 0
    best = np.zeros(n, dtype=np.int32)
    for i in range(n - 1, -1, -1):
        succ = np.flatnonzero(adj[i])
        if succ.size:
            best[i] = 1 + int(best[succ].max())
    return int(best.max())


def tag_length_histogram(adj: np.ndarray) -> dict:
    n = adj.shape[0]
    if n == 0:
        return {"n_tags_ge2": 0, "n_tags_ge3": 0, "n_tags_ge4": 0}
    best = np.zeros(n, dtype=np.int32)
    for i in range(n - 1, -1, -1):
        succ = np.flatnonzero(adj[i])
        if succ.size:
            best[i] = 1 + int(best[succ].max())
    return {
        "n_tags_ge2": int((best >= 2).sum()),
        "n_tags_ge3": int((best >= 3).sum()),
        "n_tags_ge4": int((best >= 4).sum()),
    }


def complementary_pairs(mz, inten, prec_neutral_mass, tol=0.02):
    """
    Count peak pairs whose m/z sum matches a b/y complement. Also returns the
    fraction of TIC carried by peaks involved in such a pair, which is more
    informative than the raw count.
    """
    if mz.size < 2 or prec_neutral_mass <= 0:
        return 0, 0.0
    target = prec_neutral_mass + 2 * PROTON
    sums = mz[:, None] + mz[None, :]
    pair = np.triu(np.abs(sums - target) < tol, k=1)
    count = int(pair.sum())
    if count == 0:
        return 0, 0.0
    involved = pair.any(axis=0) | pair.any(axis=1)
    frac = float(inten[involved].sum() / inten.sum()) if inten.sum() > 0 else 0.0
    return count, frac


def _nearest_within(mz, targets, tol):
    """
    For each target, the index of the nearest peak within tol, else -1.
    Peaks must be sorted. Vectorised: this runs for every spectrum, so the
    obvious nested loop is not affordable.
    """
    n = mz.size
    if n == 0:
        return np.full(targets.shape, -1, dtype=np.int64)
    idx = np.searchsorted(mz, targets)
    best = np.full(targets.shape, -1, dtype=np.int64)
    best_d = np.full(targets.shape, np.inf)
    for shift in (-1, 0):
        cand = np.clip(idx + shift, 0, n - 1)
        valid = (idx + shift >= 0) & (idx + shift < n)
        d = np.abs(mz[cand] - targets)
        take = valid & (d < tol) & (d < best_d)
        best[take] = cand[take]
        best_d[take] = d[take]
    return best


def isotope_envelope_stats(mz, inten, tol=0.01, max_charge=3):
    """
    Fraction of intensity that sits in a detectable isotope cluster, and the
    number of clusters found. Poor deisotoping is a strong 'this is noise' signal.
    """
    if mz.size < 2:
        return 0, 0.0
    in_cluster = np.zeros(mz.size, dtype=bool)
    seeds = np.zeros(mz.size, dtype=bool)
    for z in range(1, max_charge + 1):
        nxt = _nearest_within(mz, mz + NEUTRON / z, tol)
        found = nxt >= 0
        # require a decreasing-ish envelope so random coincidences do not count
        ok = found.copy()
        ok[found] &= inten[nxt[found]] < inten[found] * 1.5
        seeds |= ok
        in_cluster |= ok
        in_cluster[nxt[ok]] = True
    total = inten.sum()
    frac = float(inten[in_cluster].sum() / total) if total > 0 else 0.0
    return int(seeds.sum()), frac


def polymer_ladder(mz, inten, tol=0.02, min_chain=4):
    """
    Detect a repeating constant mass difference. Returns (delta, chain_length,
    name). Polymers, detergents and siloxanes dominate the 'high quality but
    unidentified' pile and must be removed before they eat GPU time.
    """
    n = mz.size
    if n < min_chain:
        return 0.0, 0, ""
    best = (0.0, 0, "")
    # candidate deltas: observed gaps in a plausible repeat-unit range
    diffs = mz[None, :] - mz[:, None]
    cand = diffs[np.triu(np.ones_like(diffs, dtype=bool), k=1)]
    cand = cand[(cand > 30.0) & (cand < 200.0)]
    if cand.size == 0:
        return best
    # bin the candidate deltas and take the most frequent few
    hist, edges = np.histogram(cand, bins=np.arange(30.0, 200.0, 0.02))
    for bi in np.argsort(hist)[-8:][::-1]:
        if hist[bi] < min_chain - 1:
            continue
        delta = float((edges[bi] + edges[bi + 1]) / 2)
        # longest chain with this spacing, by dynamic programming over a
        # precomputed "next peak at +delta" pointer array
        nxt = _nearest_within(mz, mz + delta, tol)
        length = np.ones(n, dtype=np.int32)
        for i in range(n - 1, -1, -1):
            j = nxt[i]
            if j > i:
                length[i] = 1 + length[j]
        longest = int(length.max())
        if longest > best[1]:
            name = ""
            for known, label in POLYMER_DELTAS.items():
                if abs(known - delta) < 0.05:
                    name = label
                    break
            best = (delta, longest, name)
    return best


def neutral_loss_features(mz, inten, prec_mz, tol=0.02, charge=1):
    """Water/ammonia losses from the precursor: sample-prep and peptide signals."""
    out = {}
    total = inten.sum() if inten.size else 1.0
    for label, loss in (("h2o", H2O), ("nh3", NH3), ("phospho", 97.9769)):
        target = prec_mz - loss / max(int(charge), 1)
        hit = np.abs(mz - target) < tol
        out[f"loss_{label}_frac"] = float(inten[hit].sum() / total) if total > 0 else 0.0
    return out


# --------------------------------------------------------------------------
# precursor context
# --------------------------------------------------------------------------

def isolation_purity(ms1_mz, ms1_inten, center, lower, upper,
                     charge=2, n_iso=4, tol=0.02):
    """
    Fraction of the intensity inside the isolation window that belongs to the
    target precursor's isotope cluster. Low values mean a chimeric spectrum.
    """
    if ms1_mz is None or ms1_mz.size == 0:
        return np.nan, 0
    win = (ms1_mz >= center - lower) & (ms1_mz <= center + upper)
    total = ms1_inten[win].sum()
    if total <= 0:
        return np.nan, 0
    spacing = NEUTRON / max(charge, 1)
    win_mz, win_int = ms1_mz[win], ms1_inten[win]

    # Numerator and denominator must both be restricted to the isolation
    # window. Summing isotopes that fall outside the window into the
    # numerator inflates purity - an interfered precursor whose envelope
    # trails past the window edge would look clean.
    target = 0.0
    for k in range(-1, n_iso):
        m = center + k * spacing
        target += win_int[np.abs(win_mz - m) < tol].sum()
    # count how many distinct non-target signals sit in the window
    thresh = win_int.max() * 0.05
    others = 0
    for m, i in zip(win_mz, win_int):
        if i < thresh:
            continue
        if min(abs(m - (center + k * spacing)) for k in range(-1, n_iso)) > tol:
            others += 1
    return float(min(target / total, 1.0)), int(others)


# --------------------------------------------------------------------------
# top-level per-spectrum feature vector
# --------------------------------------------------------------------------

def spectrum_features(mz, inten, prec_mz, charge, ms1=None,
                      iso_lower=0.7, iso_upper=0.7, max_peaks=150,
                      frag_tol=0.02):
    """
    Compute the full feature dict for one MS2 spectrum.

    mz, inten : centroided peak arrays, sorted by m/z
    ms1       : optional (mz, intensity) tuple of the preceding MS1 scan
    """
    f = {}
    if inten.size == 0:
        return None

    total_tic = float(inten.sum())
    noise = estimate_noise(inten)
    above_noise = inten > (noise * 3.0)

    f["n_peaks_raw"] = int(mz.size)
    f["n_peaks_above_noise"] = int(above_noise.sum())
    f["tic"] = total_tic
    f["base_peak_intensity"] = float(inten.max())
    f["noise_floor"] = noise
    f["snr_proxy"] = float(inten.max() / noise) if noise > 0 else np.inf

    smz, sint = top_n(mz, inten, max_peaks)
    sorted_int = np.sort(sint)[::-1]
    f["entropy"] = spectral_entropy(sint)
    f["norm_entropy"] = normalised_entropy(sint)
    f["top10_frac"] = float(sorted_int[:10].sum() / sint.sum())
    f["top20_frac"] = float(sorted_int[:20].sum() / sint.sum())
    f["log_dynamic_range"] = float(
        np.log10(sint.max() / max(np.median(sint), 1e-9)))

    prec_neutral = (prec_mz - PROTON) * charge if charge else 0.0
    f["precursor_mz"] = float(prec_mz)
    f["charge"] = int(charge)
    f["precursor_neutral_mass"] = float(prec_neutral)
    f["frac_above_precursor"] = float(
        sint[smz > prec_mz].sum() / sint.sum()) if sint.sum() > 0 else 0.0

    adj = gap_adjacency(smz, frag_tol)
    f["n_residue_gaps"] = int(adj.sum())
    f["gap_density"] = float(adj.sum() / max(smz.size, 1))
    f["longest_tag"] = longest_tag(adj)
    f.update(tag_length_histogram(adj))

    npair, pair_frac = complementary_pairs(smz, sint, prec_neutral, frag_tol)
    f["n_complementary"] = npair
    f["complementary_tic_frac"] = pair_frac

    nclus, iso_frac = isotope_envelope_stats(smz, sint)
    f["n_isotope_clusters"] = nclus
    f["isotope_tic_frac"] = iso_frac

    delta, chain, name = polymer_ladder(smz, sint)
    f["ladder_delta"] = delta
    f["ladder_length"] = chain
    f["ladder_label"] = name
    f["is_polymer_like"] = bool(chain >= 5 and name != "")

    f.update(neutral_loss_features(smz, sint, prec_mz, charge=charge))

    if ms1 is not None:
        purity, others = isolation_purity(
            ms1[0], ms1[1], prec_mz, iso_lower, iso_upper, charge)
        f["isolation_purity"] = purity
        f["n_cofragmented"] = others
    else:
        f["isolation_purity"] = np.nan
        f["n_cofragmented"] = 0

    return f
