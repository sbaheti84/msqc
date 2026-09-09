"""
Theoretical b/y fragment generation, used to draw the mirror plot in the
report. Deliberately simple: this is for visual evaluation, not for scoring.

Modification syntax understood:
  PEPT[79.9663]IDE      bracketed delta mass after the residue
  n[42.0106]PEPTIDE     N-terminal delta
  C+57.02146            plus-notation
"""

from __future__ import annotations

import re

from .features import H2O, NH3, PROTON, RESIDUE_MASSES

MOD_RE = re.compile(r"([A-Znc])(?:\[([-+]?\d*\.?\d+)\]|\+(\d*\.?\d+))?")

ION_COLOURS = {"b": "b", "y": "y"}


def parse_peptide(seq: str):
    """Return (residues, per-residue delta masses, n-term delta, c-term delta)."""
    if not seq:
        return [], [], 0.0, 0.0
    seq = str(seq).strip().replace("(", "[").replace(")", "]")
    residues, deltas = [], []
    nterm = cterm = 0.0
    for m in MOD_RE.finditer(seq):
        aa = m.group(1)
        delta = float(m.group(2) or m.group(3) or 0.0)
        if aa == "n":
            nterm += delta
            continue
        if aa == "c":
            cterm += delta
            continue
        if aa not in RESIDUE_MASSES:
            continue
        residues.append(aa)
        deltas.append(delta)
    return residues, deltas, nterm, cterm


def theoretical_fragments(seq: str, max_charge: int = 2):
    """
    Return a list of dicts: {mz, label, series, index, charge}.
    b and y ions at charge 1 and 2 (capped by precursor charge - 1... but we
    just cap at 2, which covers almost everything useful for a mirror plot).
    """
    residues, deltas, nterm, cterm = parse_peptide(seq)
    n = len(residues)
    if n < 2:
        return []

    masses = [RESIDUE_MASSES[a] + d for a, d in zip(residues, deltas)]
    out = []

    # b ions: sum of the first i residues + n-terminal delta
    running = nterm
    for i in range(n - 1):
        running += masses[i]
        for z in range(1, max_charge + 1):
            out.append({
                "mz": (running + z * PROTON) / z,
                "label": f"b{i + 1}" + ("++" if z == 2 else ""),
                "series": "b", "index": i + 1, "charge": z,
            })

    # y ions: sum of the last i residues + water + c-terminal delta
    running = H2O + cterm
    for i in range(n - 1):
        running += masses[n - 1 - i]
        for z in range(1, max_charge + 1):
            out.append({
                "mz": (running + z * PROTON) / z,
                "label": f"y{i + 1}" + ("++" if z == 2 else ""),
                "series": "y", "index": i + 1, "charge": z,
            })

    return [f for f in out if 50.0 < f["mz"] < 3000.0]


def match_fragments(obs_mz, theo, tol=0.02):
    """Annotate observed peaks. Returns index -> label mapping."""
    ann = {}
    for t in theo:
        best_i, best_d = None, tol
        for i, m in enumerate(obs_mz):
            d = abs(m - t["mz"])
            if d < best_d:
                best_i, best_d = i, d
        if best_i is not None and best_i not in ann:
            ann[best_i] = t
    return ann
