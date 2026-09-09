"""
Automated PSM validation, derived from a manual CID/HCD interpretation
checklist.

Scope note. This module answers a different question from the rest of msqc.
The QC pipeline scores spectra *without* a peptide assignment, to decide what
is worth rescuing. This module takes an assignment as given and asks whether
it holds up. Most items here cannot be applied to unassigned spectra at all,
because they need a sequence.

Analyzer awareness. Fragment mass accuracy is not a single number. An
Orbitrap MS1 precursor is good to 2-5 ppm, an Orbitrap HCD fragment scan at
15-30k resolution is more like 5-20 ppm, a TOF is 5-20 ppm, and ion-trap CID
fragments are low resolution where ppm is not a meaningful unit at all
(use ~0.3-0.5 Da). Applying a single 2 ppm rule to fragments would reject
almost every correct PSM. Set --analyzer accordingly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import PROTON
from .fragments import parse_peptide, theoretical_fragments

# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

# Diagnostic / signature ions that must be present for a PTM claim to stand.
# m/z values are singly protonated immonium-type ions.
PTM_SIGNATURE_IONS = {
    ("Phospho", "Y"): [216.04257],
    ("Acetyl", "K"): [126.09189, 143.11844],
    ("Methyl", "R"): [143.1291, 115.0866, 112.0869, 74.0713, 70.0651],
    ("Methyl", "K"): [115.12352, 98.0967, 84.08132],
    ("Dimethyl", "R"): [157.1448, 115.0866, 112.0869, 88.0869, 71.0604],
    ("Dimethyl", "K"): [129.13917, 84.08132],
    ("Trimethyl", "K"): [143.15482, 84.08132],
    ("ADP-ribosyl", "EDKR"): [543.07677, 428.03725, 348.07091,
                              250.09402, 136.06232],
}

# Delta masses that are routinely mistaken for one another. Each entry lists
# what else could produce the same mass shift, and how far apart the
# alternatives are, which determines the resolving power you need to tell
# them apart.
DELTA_MASS_AMBIGUITIES = [
    # (name, delta, [alternative explanations], nearest_confusable_delta)
    ("Deamidation", 0.984016,
     ["13C isotope error - the search picked the wrong monoisotopic peak"],
     0.997035),
    ("Methyl", 14.015650,
     ["G->A", "D->E", "V->I/L", "S->T", "N->Q", "formaldehyde adduct"],
     None),
    ("Formyl", 27.994915,
     ["Dimethyl (28.031300) - only 0.036 Da away"],
     28.031300),
    ("Dimethyl", 28.031300,
     ["A->V", "C->M", "Formyl (27.994915) - only 0.036 Da away"],
     27.994915),
    ("Di-oxidation", 31.989829, ["P->E"], None),
    ("Acetyl", 42.010565,
     ["S->E", "Trimethyl (42.046950) - only 0.036 Da away"],
     42.046950),
    ("Trimethyl", 42.046950,
     ["A->L/I", "G->V", "Acetyl (42.010565) - only 0.036 Da away"],
     42.010565),
    ("Carbamyl", 43.005814,
     ["Acetyl + 1 Da - check you have the right monoisotopic peak",
      "urea in the sample buffer"], 42.010565),
    ("GG (ubiquitin remnant)", 114.042927,
     ["double carbamidomethylation (2 x 57.021464)"], None),
]

# Immonium ions, useful as residue-presence evidence.
IMMONIUM_IONS = {
    "F": 120.0808, "Y": 136.0757, "W": 159.0917, "H": 110.0713,
    "P": 70.0651, "L/I": 86.0964, "V": 72.0808, "K": 101.1073,
    "R": 129.1135, "M": 104.0528, "C": 76.0221,
}

# TMT and iTRAQ reporter regions.
REPORTER_IONS = {
    "TMT": [126.12773, 127.12476, 127.13108, 128.12811, 128.13443,
            129.13147, 129.13779, 130.13482, 130.14114, 131.13818,
            131.14450],
    "iTRAQ": [113.10788, 114.11123, 115.10826, 116.11162, 117.11497,
              118.11201, 119.11536, 121.12200],
}

# Analyzer presets: (precursor_ppm, fragment_ppm, fragment_da).
# Use fragment_da for low-resolution ion trap data, where ppm is meaningless.
ANALYZER_TOLERANCE = {
    "orbitrap_hcd": (5.0, 20.0, None),
    "orbitrap_cid_it": (5.0, None, 0.4),   # Orbitrap MS1, ion trap MS2
    "tof": (10.0, 25.0, None),
    "iontrap": (500.0, None, 0.5),
}

# Residues whose N-terminal side is preferentially cleaved, enhancing the
# y ion that starts at them, and residues that enhance b ions.
Y_ENHANCING = set("PGS")
B_ENHANCING = set("VQH")
BASIC = set("RKH")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _match(peaks_mz, target, tol_da):
    """Index of the closest peak within tolerance, or -1."""
    if peaks_mz.size == 0:
        return -1
    i = int(np.argmin(np.abs(peaks_mz - target)))
    return i if abs(peaks_mz[i] - target) <= tol_da else -1


def _tol_da(mz, frag_ppm, frag_da):
    return frag_da if frag_da is not None else mz * frag_ppm * 1e-6


def annotate_spectrum(mz, inten, peptide, precursor_charge,
                      frag_ppm=20.0, frag_da=None, max_frag_charge=2):
    """
    Match theoretical b/y ions to the peak list.

    Returns a dict with the matched fragments, per-fragment ppm errors, the
    fraction of intensity left unexplained, and which backbone bonds are
    covered. Bond coverage, not ion count, is the thing that matters: ten
    ions clustered at one end of the peptide do not localise anything.
    """
    mz = np.asarray(mz, float)
    inten = np.asarray(inten, float)
    frags = theoretical_fragments(peptide, max_charge=min(max_frag_charge,
                                                          max(precursor_charge - 1, 1)))
    residues, _, _, _ = parse_peptide(peptide)
    n_res = len(residues)

    matched, errors_ppm, used = [], [], np.zeros(mz.size, dtype=bool)
    covered_b = np.zeros(max(n_res - 1, 0), dtype=bool)
    covered_y = np.zeros(max(n_res - 1, 0), dtype=bool)

    for fr in frags:
        tol = _tol_da(fr["mz"], frag_ppm, frag_da)
        j = _match(mz, fr["mz"], tol)
        if j < 0:
            continue
        used[j] = True
        err = (mz[j] - fr["mz"]) / fr["mz"] * 1e6
        errors_ppm.append(err)
        matched.append({**fr, "obs_mz": float(mz[j]),
                        "intensity": float(inten[j]), "error_ppm": float(err)})
        # bond index i means cleavage between residue i and i+1
        bond = fr["index"] - 1
        if 0 <= bond < covered_b.size:
            (covered_b if fr["series"] == "b" else covered_y)[bond] = True

    total = inten.sum()
    covered_any = covered_b | covered_y
    return {
        "matched": matched,
        "n_matched": len(matched),
        "median_abs_error_ppm": float(np.median(np.abs(errors_ppm)))
        if errors_ppm else np.nan,
        "median_error_ppm": float(np.median(errors_ppm)) if errors_ppm else np.nan,
        "explained_tic_frac": float(inten[used].sum() / total) if total > 0 else 0.0,
        "unexplained_tic_frac": float(inten[~used].sum() / total) if total > 0 else 0.0,
        "bond_coverage": float(covered_any.mean()) if covered_any.size else 0.0,
        "longest_consecutive_series": _longest_run(covered_any),
        "n_residues": n_res,
        "unmatched_mz": mz[~used],
        "unmatched_intensity": inten[~used],
    }


def _longest_run(flags):
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return int(best)


def check_dominant_unannotated(ann, inten, top_n=10):
    """
    Checklist A4. Counts how many of the ten most intense peaks the
    assignment fails to explain. A correct PSM usually explains most of the
    base peaks; failing to do so points at co-isolation or a wrong ID.
    """
    inten = np.asarray(inten, float)
    if inten.size == 0:
        return {"n_top10_unannotated": 0, "top_unannotated_rank": None}
    thresh = np.sort(inten)[-min(top_n, inten.size)]
    um = ann["unmatched_intensity"]
    n_top = int((um >= thresh).sum())
    rank = None
    if um.size:
        order = np.sort(inten)[::-1]
        rank = int(np.searchsorted(-order, -um.max()) + 1)
    return {"n_top10_unannotated": n_top, "top_unannotated_rank": rank}


def check_ptm_signature_ions(mz, inten, peptide, tol_da=0.01):
    """
    Checklist A5. For a modified peptide, look for the diagnostic ions that
    the modification is known to produce. Absence is not proof of a wrong
    call, but presence is strong supporting evidence.
    """
    mz = np.asarray(mz, float)
    residues, deltas, nterm, cterm = parse_peptide(peptide)
    present = set(residues)
    found, expected = [], []
    for (ptm, site), ions in PTM_SIGNATURE_IONS.items():
        if not (set(site) & present):
            continue
        expected.append(ptm)
        hits = [t for t in ions if _match(mz, t, tol_da) >= 0]
        if hits:
            found.append(f"{ptm}@{site}({len(hits)}/{len(ions)})")
    return {"signature_ions_found": "; ".join(found) or None,
            "signature_ions_possible": "; ".join(expected) or None}


def check_delta_mass_ambiguity(delta_mass, tol_da=0.01):
    """
    Checklist C2 and C4. Given an observed delta mass, list every other
    explanation with the same nominal shift, and state the resolving power
    needed to separate the closest pair.

    The acetyl / trimethyl pair at 0.036385 Da apart is the classic trap.
    At m/z 1000 that separation needs roughly 27,000 resolving power, so it
    must be judged on the precursor in the MS1, never on a low-resolution
    MS2.
    """
    if delta_mass is None or not np.isfinite(delta_mass) or abs(delta_mass) < 0.01:
        return {"delta_alternatives": None, "delta_needs_resolution": None}
    out, needed = [], None
    for name, delta, alts, near in DELTA_MASS_AMBIGUITIES:
        if abs(delta_mass - delta) <= tol_da:
            out.append(f"{name}: " + "; ".join(alts))
            if near is not None:
                sep = abs(delta - near)
                needed = max(needed or 0, 1000.0 / sep)  # R needed at m/z 1000
    return {"delta_alternatives": " | ".join(out) or None,
            "delta_needs_resolution": round(needed) if needed else None}


def check_reporter_ions(mz, inten, label="TMT", tol_da=0.005):
    """Checklist A7."""
    if label not in REPORTER_IONS:
        return {"reporter_ions_found": None, "reporter_tic_frac": None}
    mz = np.asarray(mz, float)
    inten = np.asarray(inten, float)
    hit = np.zeros(mz.size, dtype=bool)
    n = 0
    for t in REPORTER_IONS[label]:
        j = _match(mz, t, tol_da)
        if j >= 0:
            hit[j] = True
            n += 1
    total = inten.sum()
    return {"reporter_ions_found": n,
            "reporter_tic_frac": float(inten[hit].sum() / total) if total > 0 else 0.0}


def check_immonium(mz, peptide, tol_da=0.01):
    """Checklist B5. Immonium ions as independent residue evidence."""
    mz = np.asarray(mz, float)
    residues, _, _, _ = parse_peptide(peptide)
    present = set(residues)
    consistent, contradicting = [], []
    for res, ion in IMMONIUM_IONS.items():
        seen = _match(mz, ion, tol_da) >= 0
        in_pep = bool(set(res.split("/")) & present)
        if seen and in_pep:
            consistent.append(res)
        elif seen and not in_pep and res in ("W", "Y", "F"):
            # aromatic immonium ions are intense and rarely spurious, so an
            # unexplained one is a genuine warning sign
            contradicting.append(res)
    return {"immonium_consistent": ",".join(consistent) or None,
            "immonium_unexplained": ",".join(contradicting) or None}


def check_cleavage_chemistry(peptide, ann):
    """
    Checklist B1, B2 and B4. CID/HCD fragmentation is not uniform. Enhanced
    cleavage N-terminal to proline is the strongest effect; basic residues
    sequester the proton and bias the series that retains them.

    This does not pass or fail a PSM. It tells you whether the intensity
    pattern you are looking at is the one chemistry predicts, which is what
    separates a plausible spectrum from a suspiciously flat one.
    """
    residues, _, _, _ = parse_peptide(peptide)
    if len(residues) < 2:
        return {}
    seq = "".join(residues)
    n_basic = sum(1 for r in seq if r in BASIC)
    b_int = sum(m["intensity"] for m in ann["matched"] if m["series"] == "b")
    y_int = sum(m["intensity"] for m in ann["matched"] if m["series"] == "y")
    tot = b_int + y_int

    # is the bond N-terminal to proline covered, and is it dominant?
    pro_bonds = [i for i, r in enumerate(seq) if r == "P" and i > 0]
    pro_covered = None
    if pro_bonds:
        pro_covered = any(
            (m["series"] == "y" and m["index"] == len(seq) - i)
            or (m["series"] == "b" and m["index"] == i)
            for i in pro_bonds for m in ann["matched"])

    return {
        "y_to_b_intensity_ratio": float(y_int / b_int) if b_int > 0 else np.inf,
        "y_fraction": float(y_int / tot) if tot > 0 else np.nan,
        "n_basic_residues": n_basic,
        "has_proline": "P" in seq,
        "proline_bond_covered": pro_covered,
        "c_term_basic": seq[-1] in "KR",
        "missed_cleavages": sum(1 for i, r in enumerate(seq[:-1])
                                if r in "KR" and seq[i + 1] != "P"),
        "missed_cleavage_explained_by_proline": any(
            r in "KR" and seq[i + 1] == "P" for i, r in enumerate(seq[:-1])),
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def verdict(row, analyzer="orbitrap_hcd"):
    """
    Roll the individual checks into flags. Deliberately returns a list of
    concerns rather than a single score: a PSM that fails one hard check is
    not the same as one that trips three soft ones, and collapsing that into
    a number hides the reason.
    """
    prec_ppm, frag_ppm, frag_da = ANALYZER_TOLERANCE[analyzer]
    fail, warn = [], []

    if row.get("bond_coverage", 0) < 0.5:
        fail.append(f"only {row['bond_coverage']:.0%} of backbone bonds covered")
    if row.get("longest_consecutive_series", 0) < 3:
        fail.append(f"longest consecutive ion series is "
                    f"{row.get('longest_consecutive_series', 0)}")
    if row.get("explained_tic_frac", 0) < 0.3:
        fail.append(f"assignment explains only "
                    f"{row['explained_tic_frac']:.0%} of the intensity")

    err = row.get("median_abs_error_ppm")
    if frag_da is None and err is not None and np.isfinite(err) and err > frag_ppm:
        warn.append(f"median fragment error {err:.1f} ppm exceeds "
                    f"{frag_ppm:.0f} ppm for {analyzer}")
    pe = row.get("precursor_error_ppm")
    if pe is not None and np.isfinite(pe) and abs(pe) > prec_ppm:
        warn.append(f"precursor error {pe:.1f} ppm exceeds {prec_ppm:.0f} ppm")

    if row.get("n_top10_unannotated", 0) >= 5:
        warn.append(f"{row['n_top10_unannotated']} of the top 10 peaks "
                    f"are unexplained")
    ip = row.get("isolation_purity")
    if ip is not None and np.isfinite(ip) and ip < 0.5:
        warn.append(f"isolation purity {ip:.2f} - spectrum is a mixture, "
                    f"unexplained peaks may belong to a co-isolated precursor")
    if row.get("delta_alternatives"):
        warn.append("delta mass has alternative explanations")
    if row.get("immonium_unexplained"):
        warn.append(f"unexplained aromatic immonium ion(s): "
                    f"{row['immonium_unexplained']}")
    if row.get("is_modified") and not row.get("signature_ions_found") \
            and row.get("signature_ions_possible"):
        warn.append("no diagnostic ion found for the claimed modification")

    return ("fail" if fail else "warn" if warn else "pass",
            "; ".join(fail + warn) or None)
