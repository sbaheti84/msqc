"""
Regression tests. Run with:  python -m pytest tests -q

The identifier-parsing tests are not academic. Every bug found while building
this pipeline was a key-matching bug: falcon writes USIs, Casanovo writes
ms_run indices, MSFragger writes native IDs, and merging on the wrong key
silently duplicates rows instead of failing. Duplicated rows then inflate
every count downstream, which is the kind of error that survives all the way
into a figure.
"""

import os
import numpy as np
import pandas as pd
import pytest

from msqc import features, psm, rescue, score


# ---------------------------------------------------------------------------
# Feature maths
# ---------------------------------------------------------------------------

def _peptide_like_spectrum(seq_masses, start=200.0, noise=0):
    """Build a synthetic b-ion ladder so tags and gaps are known exactly."""
    mz = [start]
    for m in seq_masses:
        mz.append(mz[-1] + m)
    mz = np.array(mz)
    inten = np.full(mz.size, 1000.0)
    if noise:
        rng = np.random.default_rng(0)
        nmz = rng.uniform(150, 1200, noise)
        mz = np.concatenate([mz, nmz])
        inten = np.concatenate([inten, rng.uniform(10, 80, noise)])
        order = np.argsort(mz)
        mz, inten = mz[order], inten[order]
    return mz, inten


def test_longest_tag_counts_residues_not_peaks():
    # Four residues chained -> tag length 4
    g, a, s, v = 57.02146, 71.03711, 87.03203, 99.06841
    mz, inten = _peptide_like_spectrum([g, a, s, v])
    f = features.spectrum_features(mz, inten, prec_mz=800.0, charge=2)
    assert f["longest_tag"] == 4


def test_pure_noise_has_no_tag():
    rng = np.random.default_rng(42)
    mz = np.sort(rng.uniform(150, 1200, 60))
    inten = rng.uniform(10, 100, 60)
    f = features.spectrum_features(mz, inten, prec_mz=700.0, charge=2)
    # Random peaks occasionally chain by chance; a real tag should not appear
    assert f["longest_tag"] <= 2


def test_polymer_ladder_is_detected():
    # Evenly spaced PEG ladder: 44.0262 Da repeats
    mz = np.array([300.0 + 44.026215 * i for i in range(10)])
    inten = np.full(mz.size, 5000.0)
    f = features.spectrum_features(mz, inten, prec_mz=600.0, charge=2)
    assert f["ladder_length"] >= 8
    assert f["is_polymer_like"]


def test_peptide_is_not_flagged_as_polymer():
    g, a, s, v, t, p = 57.02146, 71.03711, 87.03203, 99.06841, 101.04768, 97.05276
    mz, inten = _peptide_like_spectrum([g, a, s, v, t, p], noise=20)
    f = features.spectrum_features(mz, inten, prec_mz=800.0, charge=2)
    assert not f["is_polymer_like"]


def test_isolation_purity_detects_cofragmentation():
    # Clean: only the target isotope envelope inside the window
    ms1_mz = np.array([500.00, 500.50, 501.00, 501.50])
    ms1_int = np.array([1e6, 5e5, 2e5, 5e4])
    clean = features.isolation_purity(ms1_mz, ms1_int, 500.0, 0.7, 0.7, 2)
    # Dirty: an interfering species of equal intensity in the same window
    ms1_mz2 = np.append(ms1_mz, 500.31)
    ms1_int2 = np.append(ms1_int, 1e6)
    dirty = features.isolation_purity(ms1_mz2, ms1_int2, 500.0, 0.7, 0.7, 2)
    assert clean[0] == pytest.approx(1.0)
    # 1.5e6 of 2.5e6 in-window intensity belongs to the target envelope
    assert dirty[0] == pytest.approx(0.6, abs=0.01)
    assert dirty[1] >= 1  # the interfering species is counted


def test_empty_spectrum_returns_none_not_a_crash():
    assert features.spectrum_features(np.array([]), np.array([]),
                                      prec_mz=500.0, charge=0) is None


def test_sparse_spectra_still_produce_a_full_feature_vector():
    """One- and two-peak spectra are common at the end of a gradient. They
    must yield the same keys as any other spectrum or the Parquet schema
    changes halfway through a run."""
    keys = None
    for mz, inten in [(np.array([500.0]), np.array([1.0])),
                      (np.array([500.0, 600.0]), np.array([1.0, 2.0]))]:
        f = features.spectrum_features(mz, inten, prec_mz=500.0, charge=0)
        assert isinstance(f, dict)
        if keys is None:
            keys = set(f)
        assert set(f) == keys


# ---------------------------------------------------------------------------
# Identifier parsing: every format these tools actually emit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("identifier,run,scan", [
    ("mzspec:PXD000561:file_A.mzML:scan:1234", "file_A", 1234),
    ("demo_run_01.4567", "demo_run_01", 4567),
    ("/data/runs/sample_B.mzML:scan:77", "sample_B", 77),
    ("sample_C.mgf.99", "sample_C", 99),
    ("run_D.mzML index=42", "run_D", 42),
])
def test_spectrum_identifier_formats(identifier, run, scan):
    out = psm.parse_spectrum_identifier(pd.Series([identifier]))
    assert out.loc[0, "run_id"] == run
    assert out.loc[0, "scan_number"] == scan


def test_unparseable_identifier_yields_nan_not_a_guess():
    out = psm.parse_spectrum_identifier(pd.Series(["no-scan-here"]))
    assert pd.isna(out.loc[0, "scan_number"])


def test_casanovo_mztab_resolves_ms_run_to_filenames(tmp_path):
    """
    The bug this guards: without resolving ms_run[N] to a filename, a
    two-run mzTab merges on scan number alone and duplicates rows.
    """
    p = tmp_path / "c.mztab"
    p.write_text(
        "MTD\tmzTab-version\t1.0.0\n"
        "MTD\tms_run[1]-location\tfile:///x/runA.mzML\n"
        "MTD\tms_run[2]-location\tfile:///x/runB.mzML\n"
        "PSH\tsequence\tPSM_ID\tsearch_engine_score[1]\tspectra_ref\n"
        "PSM\tPEPTIDEK\t0\t0.91\tms_run[1]:scan=10\n"
        "PSM\tSEQUENCER\t1\t0.72\tms_run[2]:scan=10\n")
    df = psm.read_casanovo_mztab(str(p))
    assert set(df["run_id"]) == {"runA", "runB"}
    assert df.duplicated(["run_id", "scan_number"]).sum() == 0


def test_annotate_merge_does_not_inflate_rows(tmp_path):
    qc = pd.DataFrame({
        "run_id": ["runA", "runA", "runB", "runB"],
        "scan_number": [10, 11, 10, 11],
    })
    p = tmp_path / "f.csv"
    pd.DataFrame({
        "identifier": ["mzspec:D:runA.mzML:scan:10",
                       "mzspec:D:runB.mzML:scan:10"],
        "cluster": [0, 0],
    }).to_csv(p, index=False)
    cl = psm.read_clusters(str(p))
    merged = qc.merge(cl, on=["run_id", "scan_number"], how="left")
    assert len(merged) == len(qc)
    assert merged["cluster_size"].max() == 2
    assert merged["cluster_n_runs"].max() == 2


def test_unclustered_spectra_are_dropped(tmp_path):
    p = tmp_path / "f.csv"
    pd.DataFrame({"identifier": ["runA.1", "runA.2"],
                  "cluster": [-1, 3]}).to_csv(p, index=False)
    cl = psm.read_clusters(str(p))
    assert list(cl["scan_number"]) == [2]


# ---------------------------------------------------------------------------
# Triage logic
# ---------------------------------------------------------------------------

def _triage_frame():
    return pd.DataFrame({
        "assigned":        [True,  False, False, False, False],
        "qc_score":        [0.9,   0.9,   0.9,   0.2,   0.95],
        "longest_tag":     [5,     5,     1,     1,     6],
        "is_polymer_like": [False, False, False, False, True],
    })


def test_triage_buckets_are_mutually_exclusive_and_complete():
    out = rescue.triage(_triage_frame())
    assert list(out["triage_class"]) == [
        "identified",
        "rescue_candidate",
        "structured_non_peptide",
        "low_quality_unassigned",
        "polymer_contaminant",
    ]


def test_polymers_never_reach_the_rescue_queue():
    """A PEG ladder scores well on every peptide-agnostic metric. If this
    test fails, the GPU queue fills with detergent."""
    out = rescue.triage(_triage_frame())
    assert not out.loc[out["is_polymer_like"], "is_rescue_candidate"].any()


def test_labels_hold_out_strong_unassigned_spectra():
    """
    The circularity guard. Structurally strong but unassigned spectra are
    exactly the rescue targets; labelling them negative teaches the model to
    throw them away.
    """
    df = pd.DataFrame({
        "assigned":        [True, False, False],
        "longest_tag":     [5,    5,     0],
        "n_complementary": [4,    4,     0],
    })
    labels = score.build_labels(df)
    assert list(labels) == [1, -1, 0]


def test_rule_score_is_bounded_and_ordered():
    df = pd.DataFrame({
        "longest_tag": [0, 3, 8], "n_tags_ge3": [0, 2, 8],
        "n_complementary": [0, 2, 10], "complementary_tic_frac": [0, .1, .5],
        "isotope_tic_frac": [0, .2, .8], "n_peaks_above_noise": [3, 20, 80],
        "norm_entropy": [1.0, .8, .75], "log_dynamic_range": [0, 1, 3],
        "isolation_purity": [.3, .7, 1.0],
    })
    s = score.rule_score(df)
    assert s.between(0, 1).all()
    assert s.is_monotonic_increasing


# ---------------------------------------------------------------------------
# Interpretation checklist
# ---------------------------------------------------------------------------

from msqc import checklist as ck


def test_fragment_tolerance_is_analyzer_specific():
    """
    The guard against the checklist's one factual error. Fragment ions are
    not held to the precursor's 2 ppm, and ion-trap MS2 is not a ppm
    instrument at all.
    """
    prec, frag_ppm, frag_da = ck.ANALYZER_TOLERANCE["orbitrap_hcd"]
    assert frag_ppm > prec            # fragments are looser than precursors
    _, ppm_it, da_it = ck.ANALYZER_TOLERANCE["orbitrap_cid_it"]
    assert ppm_it is None and da_it >= 0.3   # low-res MS2 uses Da, not ppm


def test_acetyl_trimethyl_ambiguity_is_flagged_with_required_resolution():
    r = ck.check_delta_mass_ambiguity(42.010565)
    assert "Trimethyl" in r["delta_alternatives"]
    # 0.036385 Da apart at m/z 1000 needs roughly 27,000 resolving power
    assert 25000 < r["delta_needs_resolution"] < 30000


def test_deamidation_is_flagged_against_isotope_error():
    """The most common false PTM: +0.98402 vs a 13C peak at +0.99703."""
    r = ck.check_delta_mass_ambiguity(0.984016)
    assert "isotope" in r["delta_alternatives"].lower()


def test_unmodified_delta_mass_raises_nothing():
    assert ck.check_delta_mass_ambiguity(0.0)["delta_alternatives"] is None


def test_bond_coverage_beats_ion_count():
    """
    Ten ions clustered at one terminus localise nothing. A verdict driven by
    ion count instead of bond coverage would pass this.
    """
    row = {"bond_coverage": 0.2, "longest_consecutive_series": 8,
           "explained_tic_frac": 0.9, "n_matched": 10}
    v, why = ck.verdict(row)
    assert v == "fail" and "bond" in why


def test_clean_psm_passes():
    row = {"bond_coverage": 0.9, "longest_consecutive_series": 6,
           "explained_tic_frac": 0.75, "median_abs_error_ppm": 3.0,
           "isolation_purity": 0.95, "n_top10_unannotated": 1}
    assert ck.verdict(row)[0] == "pass"


def test_chimeric_psm_is_warned_not_failed():
    """Co-isolation explains unassigned peaks; it does not make the ID wrong."""
    row = {"bond_coverage": 0.9, "longest_consecutive_series": 6,
           "explained_tic_frac": 0.6, "isolation_purity": 0.3,
           "n_top10_unannotated": 6}
    v, why = ck.verdict(row)
    assert v == "warn" and "purity" in why


# ---------------------------------------------------------------------------
# Truncated and corrupt input
# ---------------------------------------------------------------------------

from msqc import extract as ex


def _write_truncated(tmp_path, src_bytes, keep_frac=0.5):
    p = tmp_path / "trunc.mzML"
    p.write_bytes(src_bytes[:int(len(src_bytes) * keep_frac)])
    return str(p)


def test_complete_mzml_is_recognised(tmp_path):
    p = tmp_path / "ok.mzML"
    p.write_text("<indexedmzML><mzML>...</mzML></indexedmzML>\n")
    r = ex.check_truncation(str(p))
    assert r["complete"] and r["has_index"]


def test_truncated_mzml_is_detected_from_the_tail(tmp_path):
    """
    One seek to the end is enough. Without this the file parses happily for
    a million lines and then throws deep inside the run, which reads like a
    bug in the reader rather than a bad input.
    """
    p = tmp_path / "bad.mzML"
    p.write_text('<indexedmzML><mzML><run><spectrum><binary>AAAA')
    r = ex.check_truncation(str(p))
    assert not r["complete"]
    assert "closing </mzML>" in r["detail"]


def test_missing_file_reports_rather_than_raises(tmp_path):
    r = ex.check_truncation(str(tmp_path / "nope.mzML"))
    assert not r["complete"] and "could not read" in r["detail"]


def test_salvage_keeps_spectra_read_before_the_tear(tmp_path):
    """
    A file torn at 90% still holds 90% of a usable run. Losing it to an
    exception helps nobody, so long as the caller is told it is partial.
    """
    src = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "demo_data", "demo_run_01.mzML")
    if not os.path.exists(src):
        pytest.skip("demo data not generated")
    path = _write_truncated(tmp_path, open(src, "rb").read(), 0.5)
    df, warn = ex.extract_run_safe(path)
    assert len(df) > 0
    assert warn is not None and "INCOMPLETE" in warn


# ---------------------------------------------------------------------------
# RAW conversion
# ---------------------------------------------------------------------------

from msqc import convert as cv


def test_raw_and_mzml_are_told_apart():
    assert cv.is_raw("/data/Hela.raw") and cv.is_raw("/data/Hela.RAW")
    assert cv.is_mzml("/data/Hela.mzML") and not cv.is_raw("/data/Hela.mzML")


def test_mzml_inputs_pass_through_untouched(tmp_path):
    p = tmp_path / "a.mzML"
    p.write_text("x")
    out, conversions = cv.ensure_mzml([str(p)], str(tmp_path / "w"))
    assert out == [str(p)] and conversions == {}


def test_missing_converter_gives_installation_advice(tmp_path, monkeypatch):
    monkeypatch.setattr(cv, "best_backend", lambda: None)
    raw = tmp_path / "x.raw"
    raw.write_text("x")
    with pytest.raises(RuntimeError, match="thermorawfileparser"):
        cv.convert_raw(str(raw), str(tmp_path / "out"))


@pytest.mark.parametrize("backend", ["trfp_native", "trfp_docker",
                                     "msconvert_docker"])
def test_conversion_always_requests_centroided_output(backend, monkeypatch):
    """
    Profile-mode data makes every peak-count and noise feature in this
    package meaningless, so peak picking must never be optional.
    """
    monkeypatch.setattr(cv, "available_backends",
                        lambda: [{"id": backend, "ok": True,
                                  "exe": "/bin/x", "dll": "/x.exe"}])
    cmd = " ".join(cv._build_command(backend, "/d/H.raw", "/o"))
    assert "--noPeakPicking" not in cmd          # TRFP centroids by default
    if "msconvert" in backend:
        assert "peakPicking" in cmd
    else:
        assert "-f 2" in cmd                      # indexed mzML


def test_docker_command_mounts_read_only_input(monkeypatch):
    monkeypatch.setattr(cv, "available_backends",
                        lambda: [{"id": "trfp_docker", "ok": True}])
    cmd = cv._build_command("trfp_docker", "/data/H.raw", "/out")
    assert "/data:/in:ro" in cmd
    assert cv.TRFP_IMAGE in cmd and ":latest" not in " ".join(cmd)
