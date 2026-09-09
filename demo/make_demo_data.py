"""
Generate a small synthetic mzML plus a matching FragPipe-style psm.tsv so the
pipeline can be exercised end to end without real data.

The simulated run contains five populations on purpose, so you can check that
triage separates them:

  1. clean tryptic peptides that the "search" identifies
  2. clean peptides deliberately left OUT of the psm.tsv  -> should land in
     rescue_candidate
  3. modified peptides (a mass shift on one residue) left out of the psm.tsv
     -> should also land in rescue_candidate
  4. PEG polymer ladders -> should land in polymer_contaminant
  5. pure noise -> should land in low_quality_unassigned

This is a plumbing test, not a benchmark. Real spectra are messier.
"""

from __future__ import annotations

import base64
import os
import random
import zlib

import numpy as np

PROTON = 1.00727646688
H2O = 18.010565
AA = {
    "G": 57.02146, "A": 71.03711, "S": 87.03203, "P": 97.05276,
    "V": 99.06841, "T": 101.04768, "C": 103.00919, "L": 113.08406,
    "N": 114.04293, "D": 115.02694, "Q": 128.05858, "K": 128.09496,
    "E": 129.04259, "M": 131.04049, "H": 137.05891, "F": 147.06841,
    "R": 156.10111, "Y": 163.06333, "W": 186.07931,
}
ALPHABET = list(AA)


def random_peptide(rng, lo=7, hi=16):
    n = rng.integers(lo, hi)
    core = "".join(rng.choice(ALPHABET[:-3], size=n - 1))
    return core + rng.choice(["K", "R"])


def peptide_mass(seq, mods=None):
    m = H2O + sum(AA[a] for a in seq)
    if mods:
        m += sum(mods.values())
    return m


def fragment_spectrum(seq, charge, rng, mods=None, completeness=0.75,
                      noise_peaks=40, noise_level=0.05):
    """Generate a plausible b/y spectrum with missing ions and noise."""
    mods = mods or {}
    masses = [AA[a] + mods.get(i, 0.0) for i, a in enumerate(seq)]
    n = len(seq)
    peaks = []

    run = 0.0
    for i in range(n - 1):
        run += masses[i]
        if rng.random() < completeness:
            peaks.append((run + PROTON, rng.uniform(0.1, 0.6)))
    run = H2O
    for i in range(n - 1):
        run += masses[n - 1 - i]
        if rng.random() < completeness + 0.1:  # y ions usually stronger
            peaks.append((run + PROTON, rng.uniform(0.2, 1.0)))

    # a few doubly charged fragments for longer peptides
    if charge >= 3:
        for mz, inten in list(peaks[: n // 2]):
            if rng.random() < 0.2:
                peaks.append(((mz + PROTON) / 2, inten * 0.3))

    # isotope peaks
    for mz, inten in list(peaks):
        if rng.random() < 0.6:
            peaks.append((mz + 1.00335, inten * rng.uniform(0.2, 0.45)))

    for _ in range(noise_peaks):
        peaks.append((rng.uniform(150, 1400), rng.uniform(0.005, noise_level)))

    peaks = [(mz + rng.normal(0, 0.004), max(i, 1e-4)) for mz, i in peaks]
    peaks.sort()
    mzs = np.array([p[0] for p in peaks])
    ints = np.array([p[1] for p in peaks]) * rng.uniform(1e4, 5e5)
    return mzs, ints


def polymer_spectrum(rng, delta=44.0262, n_units=14):
    start = rng.uniform(300, 500)
    mzs, ints = [], []
    for k in range(n_units):
        m = start + k * delta
        mzs.append(m + rng.normal(0, 0.003))
        ints.append(np.exp(-((k - n_units / 3) ** 2) / 18) * rng.uniform(0.6, 1.0))
        if rng.random() < 0.8:
            mzs.append(m + 1.00335)
            ints.append(ints[-1] * 0.35)
    for _ in range(25):
        mzs.append(rng.uniform(200, 1200))
        ints.append(rng.uniform(0.005, 0.04))
    order = np.argsort(mzs)
    return np.array(mzs)[order], np.array(ints)[order] * 3e5


def noise_spectrum(rng, n=60):
    mzs = np.sort(rng.uniform(150, 1500, n))
    ints = rng.uniform(0.005, 0.08, n) * 2e4
    return mzs, ints


def b64(arr):
    return base64.b64encode(zlib.compress(
        np.asarray(arr, dtype="<f8").tobytes())).decode()


def binary_array(arr, kind):
    cv = ('<cvParam cvRef="MS" accession="MS:1000514" name="m/z array" '
          'unitAccession="MS:1000040" unitName="m/z" unitCvRef="MS"/>'
          if kind == "mz" else
          '<cvParam cvRef="MS" accession="MS:1000515" name="intensity array" '
          'unitAccession="MS:1000131" unitName="number of detector counts" '
          'unitCvRef="MS"/>')
    enc = b64(arr)
    return f"""        <binaryDataArray encodedLength="{len(enc)}">
          <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>
          <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression"/>
          {cv}
          <binary>{enc}</binary>
        </binaryDataArray>"""


def ms1_xml(index, scan, rt, mzs, ints):
    return f"""      <spectrum index="{index}" id="controllerType=0 controllerNumber=1 scan={scan}" defaultArrayLength="{len(mzs)}">
        <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>
        <cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum"/>
        <scanList count="1">
          <cvParam cvRef="MS" accession="MS:1000795" name="no combination"/>
          <scan>
            <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt:.4f}" unitAccession="UO:0000031" unitName="minute" unitCvRef="UO"/>
            <cvParam cvRef="MS" accession="MS:1000927" name="ion injection time" value="20.0" unitAccession="UO:0000028" unitName="millisecond" unitCvRef="UO"/>
          </scan>
        </scanList>
        <binaryDataArrayList count="2">
{binary_array(mzs, "mz")}
{binary_array(ints, "int")}
        </binaryDataArrayList>
      </spectrum>"""


def ms2_xml(index, scan, rt, prec_mz, charge, prec_int, inject, mzs, ints):
    return f"""      <spectrum index="{index}" id="controllerType=0 controllerNumber=1 scan={scan}" defaultArrayLength="{len(mzs)}">
        <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="2"/>
        <cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum"/>
        <scanList count="1">
          <cvParam cvRef="MS" accession="MS:1000795" name="no combination"/>
          <scan>
            <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt:.4f}" unitAccession="UO:0000031" unitName="minute" unitCvRef="UO"/>
            <cvParam cvRef="MS" accession="MS:1000927" name="ion injection time" value="{inject:.1f}" unitAccession="UO:0000028" unitName="millisecond" unitCvRef="UO"/>
          </scan>
        </scanList>
        <precursorList count="1">
          <precursor>
            <isolationWindow>
              <cvParam cvRef="MS" accession="MS:1000827" name="isolation window target m/z" value="{prec_mz:.5f}"/>
              <cvParam cvRef="MS" accession="MS:1000828" name="isolation window lower offset" value="0.7"/>
              <cvParam cvRef="MS" accession="MS:1000829" name="isolation window upper offset" value="0.7"/>
            </isolationWindow>
            <selectedIonList count="1">
              <selectedIon>
                <cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="{prec_mz:.5f}"/>
                <cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="{charge}"/>
                <cvParam cvRef="MS" accession="MS:1000042" name="peak intensity" value="{prec_int:.1f}"/>
              </selectedIon>
            </selectedIonList>
            <activation>
              <cvParam cvRef="MS" accession="MS:1000133" name="collision-induced dissociation"/>
            </activation>
          </precursor>
        </precursorList>
        <binaryDataArrayList count="2">
{binary_array(mzs, "mz")}
{binary_array(ints, "int")}
        </binaryDataArrayList>
      </spectrum>"""


def isotope_cluster(mz, charge, intensity, n=4, decay=0.65):
    out = []
    for k in range(n):
        out.append((mz + k * 1.00335 / charge, intensity * (decay ** k)))
    return out


def build(out_dir="demo_data", run_id="demo_run_01", n_ms2=600, seed=7,
          ms1_every=8):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # composition of the simulated run
    plan = (["identified"] * int(n_ms2 * 0.32)
            + ["unassigned_clean"] * int(n_ms2 * 0.10)
            + ["unassigned_modified"] * int(n_ms2 * 0.08)
            + ["polymer"] * int(n_ms2 * 0.08)
            + ["noise"] * (n_ms2 - int(n_ms2 * 0.32) - int(n_ms2 * 0.10)
                           - int(n_ms2 * 0.08) - int(n_ms2 * 0.08)))
    rng.shuffle(plan)

    # ---- pass 1: build every MS2 event so the MS1 can contain its precursor
    events = []
    for kind in plan:
        charge = int(rng.choice([2, 2, 2, 3, 3, 4]))
        chimeric = bool(rng.random() < 0.22)

        if kind in ("identified", "unassigned_clean", "unassigned_modified"):
            seq = random_peptide(rng)
            mods = {}
            if kind == "unassigned_modified":
                pos = int(rng.integers(0, len(seq)))
                mods[pos] = float(rng.choice([79.96633, 42.01057, 114.04293,
                                              -17.02655, 176.0321]))
            mass = peptide_mass(seq, mods)
            prec_mz = (mass + charge * PROTON) / charge
            completeness = 0.82 if kind == "identified" else rng.uniform(0.6, 0.85)
            mzs, ints = fragment_spectrum(seq, charge, rng, mods, completeness)
            prec_int = float(rng.uniform(1e5, 5e7))

            if chimeric:
                # co-isolated second peptide leaks fragments into the MS2
                other = random_peptide(rng)
                omz, oint = fragment_spectrum(other, 2, rng,
                                              completeness=0.5, noise_peaks=5)
                mzs = np.concatenate([mzs, omz])
                ints = np.concatenate([ints, oint * 0.35])
                order = np.argsort(mzs)
                mzs, ints = mzs[order], ints[order]
        elif kind == "polymer":
            seq, mods = None, {}
            mzs, ints = polymer_spectrum(rng)
            prec_mz = float(rng.uniform(500, 900))
            prec_int = float(rng.uniform(1e6, 1e7))
        else:
            seq, mods = None, {}
            mzs, ints = noise_spectrum(rng)
            prec_mz = float(rng.uniform(400, 1000))
            prec_int = float(rng.uniform(1e3, 5e4))

        events.append({
            "kind": kind, "seq": seq, "mods": mods, "charge": charge,
            "prec_mz": prec_mz, "prec_int": prec_int, "chimeric": chimeric,
            "mzs": mzs, "ints": ints,
            "inject": float(rng.uniform(8, 110)),
        })

    # ---- pass 2: emit MS1 survey scans followed by their MS2 events
    spectra_xml, psm_rows = [], []
    scan = index = 0
    rt = 5.0

    for start in range(0, len(events), ms1_every):
        block = events[start:start + ms1_every]
        rt += rng.uniform(0.005, 0.02)
        scan += 1

        peaks = []
        for ev in block:
            peaks += isotope_cluster(ev["prec_mz"], ev["charge"], ev["prec_int"])
            if ev["chimeric"]:
                # an interfering species inside the same isolation window
                off = float(rng.uniform(-0.55, 0.55))
                peaks += isotope_cluster(ev["prec_mz"] + off, ev["charge"],
                                         ev["prec_int"] * rng.uniform(0.4, 2.5))
        for _ in range(150):  # chemical background
            m = float(rng.uniform(350, 1200))
            peaks += isotope_cluster(m, 1, float(rng.uniform(5e3, 2e5)), n=2)

        peaks.sort()
        ms1_mz = np.array([p[0] for p in peaks])
        ms1_int = np.array([p[1] for p in peaks])
        spectra_xml.append(ms1_xml(index, scan, rt, ms1_mz, ms1_int))
        index += 1

        for ev in block:
            rt += rng.uniform(0.003, 0.01)
            scan += 1
            spectra_xml.append(ms2_xml(
                index, scan, rt, ev["prec_mz"], ev["charge"],
                ev["prec_int"], ev["inject"], ev["mzs"], ev["ints"]))
            index += 1

            if ev["kind"] == "identified":
                psm_rows.append({
                    "Spectrum": f"{run_id}.{scan:05d}.{scan:05d}.{ev['charge']}",
                    "Peptide": ev["seq"],
                    "Modified Peptide": ev["seq"],
                    "Charge": ev["charge"],
                    "Retention": round(rt * 60, 2),
                    "Hyperscore": round(rng.uniform(22, 48), 3),
                    "Nextscore": round(rng.uniform(5, 20), 3),
                    "Expectation": f"{rng.uniform(1e-12, 1e-4):.3e}",
                    "Delta Mass": round(rng.normal(0, 0.002), 5),
                    "Protein": f"sp|P{rng.integers(10000, 99999)}|DEMO_HUMAN",
                })

    n_spectra = index

    # ground truth, so you can check whether triage actually separates classes
    truth_rows = []
    scan = 0
    for start in range(0, len(events), ms1_every):
        scan += 1
        for ev in events[start:start + ms1_every]:
            scan += 1
            truth_rows.append({"run_id": run_id, "scan_number": scan,
                               "true_class": ev["kind"],
                               "chimeric": ev["chimeric"],
                               "true_peptide": ev["seq"] or ""})

    body = "\n".join(spectra_xml)
    mzml = f"""<?xml version="1.0" encoding="utf-8"?>
<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0" id="{run_id}">
  <cvList count="2">
    <cv id="MS" fullName="Proteomics Standards Initiative Mass Spectrometry Ontology" URI="https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"/>
    <cv id="UO" fullName="Unit Ontology" URI="https://raw.githubusercontent.com/bio-ontology-research-group/unit-ontology/master/unit.obo"/>
  </cvList>
  <fileDescription><fileContent>
    <cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum"/>
  </fileContent></fileDescription>
  <softwareList count="1"><software id="msqc_demo" version="0.1.0">
    <cvParam cvRef="MS" accession="MS:1000799" name="custom unreleased software tool" value="msqc demo generator"/>
  </software></softwareList>
  <instrumentConfigurationList count="1">
    <instrumentConfiguration id="IC1">
      <cvParam cvRef="MS" accession="MS:1000483" name="Thermo Fisher Scientific instrument model"/>
    </instrumentConfiguration>
  </instrumentConfigurationList>
  <dataProcessingList count="1"><dataProcessing id="dp1">
    <processingMethod order="0" softwareRef="msqc_demo">
      <cvParam cvRef="MS" accession="MS:1000544" name="Conversion to mzML"/>
    </processingMethod>
  </dataProcessing></dataProcessingList>
  <run id="{run_id}" defaultInstrumentConfigurationRef="IC1">
    <spectrumList count="{n_spectra}" defaultDataProcessingRef="dp1">
{body}
    </spectrumList>
  </run>
</mzML>
</indexedmzML>
"""
    mzml_path = os.path.join(out_dir, f"{run_id}.mzML")
    with open(mzml_path, "w") as fh:
        fh.write(mzml)

    import csv
    psm_path = os.path.join(out_dir, f"{run_id}.psm.tsv")
    fields = ["Spectrum", "Peptide", "Modified Peptide", "Charge", "Retention",
              "Hyperscore", "Nextscore", "Expectation", "Delta Mass", "Protein"]
    with open(psm_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(psm_rows)

    truth_path = os.path.join(out_dir, f"{run_id}.truth.tsv")
    with open(truth_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_id", "scan_number",
                                           "true_class", "chimeric",
                                           "true_peptide"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(truth_rows)

    print(f"wrote {mzml_path}  ({n_spectra} spectra, "
          f"{os.path.getsize(mzml_path) / 1e6:.1f} MB)")
    print(f"wrote {psm_path}  ({len(psm_rows)} PSMs)")
    print(f"wrote {truth_path}  ({len(truth_rows)} MS2 scans, simulated labels)")
    return mzml_path, psm_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Generate synthetic mzML + psm.tsv + ground-truth labels "
                    "so the pipeline can be exercised end to end.")
    ap.add_argument("--out", default="demo_data",
                    help="output directory (default: demo_data)")
    ap.add_argument("--runs", type=int, default=2,
                    help="how many runs to simulate. Two or more is needed "
                         "for cross-run cluster evidence and for the "
                         "group-aware train/validation split.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    for i in range(1, a.runs + 1):
        build(a.out, run_id=f"demo_run_{i:02d}", seed=a.seed + i)
