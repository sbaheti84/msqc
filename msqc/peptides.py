"""Normalize exact FragPipe modification masses for fragment annotation."""
import re
import numpy as np
from .features import RESIDUE_MASSES, H2O


def fragpipe_sequence(peptide, modified, assigned, calculated_mass=np.nan):
    if not isinstance(peptide,str) or not peptide:
        return '', 'Missing peptide sequence.'
    deltas = [0.] * len(peptide)
    nt = ct = 0.
    if isinstance(assigned,str) and assigned.strip():
        for item in assigned.split(','):
            item = item.strip()
            m = re.fullmatch(r'(\d+)([A-Z])\(([-+]?\d+(?:\.\d+)?)\)',item)
            term = re.fullmatch(r'(N-term|C-term)\(([-+]?\d+(?:\.\d+)?)\)',item,re.I)
            if m:
                pos = int(m[1])-1
                if not 0 <= pos < len(peptide) or peptide[pos] != m[2]:
                    return '', 'Modification position disagrees with peptide.'
                deltas[pos] += float(m[3])
            elif term:
                if term[1].lower() == 'n-term': nt += float(term[2])
                else: ct += float(term[2])
            else:
                return '', f'Unsupported assigned modification: {item}'
    normalized = (f'n[{nt:+.6f}]' if nt else '') + ''.join(
        a+(f'[{d:+.6f}]' if d else '') for a,d in zip(peptide,deltas)) + (f'c[{ct:+.6f}]' if ct else '')
    # FragPipe brackets contain rounded total residue masses, NOT mass deltas.
    # Never use those rounded values for high-resolution fragment matching.
    if isinstance(modified,str) and modified.strip():
        tokens = list(re.finditer(r'([A-Znc])(?:\[([-+]?\d+(?:\.\d+)?)\])?',modified))
        if ''.join(t.group(0) for t in tokens) != modified:
            return '', 'Unsupported FragPipe modified sequence syntax.'
        if ''.join(t[1] for t in tokens if t[1] not in 'nc') != peptide:
            return '', 'Modified sequence disagrees with peptide.'
        pos = 0
        for token in tokens:
            aa, mass = token[1], token[2]
            if aa in 'nc':
                # Terminal notation varies; rely on exact assigned modifications.
                if mass is not None and not (nt if aa == 'n' else ct):
                    return '', 'Exact terminal modification mass is missing.'
                continue
            if mass is not None and abs(float(mass)-(RESIDUE_MASSES.get(aa,0)+deltas[pos])) > .51:
                return '', 'Exact modification masses (including fixed modifications) are missing; rounded FragPipe masses cannot be annotated accurately.'
            pos += 1
    if any(a not in RESIDUE_MASSES for a in peptide):
        return '', 'Unsupported amino-acid residue.'
    calculated = sum(RESIDUE_MASSES[a]+d for a,d in zip(peptide,deltas))+H2O+nt+ct
    if np.isfinite(calculated_mass) and abs(calculated-calculated_mass) > .05:
        return '', 'Sequence mass differs from reported peptide mass; supply complete exact modification assignments.'
    return normalized, ''
