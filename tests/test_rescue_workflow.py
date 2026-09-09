"""Spectrum identity, chemistry, and real background-process regression tests.

External adapters are tested with controlled executable fixtures, not claimed
as scientific validation of Falcon, Casanovo, or FragPipe.
"""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pytest
from msqc import clustering, features, fragments, identity, jobs, psm, rescue, score
from msqc.checklist import annotate_spectrum
from msqc.peptides import fragpipe_sequence


@pytest.fixture
def spectra():
    peaks = np.array([110.,180.,260.,350.,420.,570.,650.,780.])
    rows = []
    for run,scan,mz,offset in [('run A',99,500.,0.),('run B',99,500.001,.002),('run B',100,750.,35.)]:
        f = features.spectrum_features(peaks+offset,np.arange(1.,9.)*100.,mz,2)
        rows.append({**f,'run_id':run,'scan_number':scan,'rt_min':10.,'charge_imputed':False,
                     '_mz':peaks+offset,'_intensity':np.arange(1.,9.)*100.,'qc_score':.95,
                     'assigned':False,'triage_class':'rescue_candidate','is_rescue_candidate':True})
    return pd.DataFrame(rows)


def write_mztab(path, refs):
    path.write_text('MTD\tms_run[1]-location\tfile:///data/representatives.mgf\n'
                    'PSH\tsequence\tsearch_engine_score[1]\tspectra_ref\n' +
                    ''.join(f'PSM\t{seq}\t{value}\tms_run[1]:{ref}\n' for seq,value,ref in refs))


def wait_job(path, timeout=30):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        state = jobs.status(path)
        if state['state'] in jobs.TERMINAL:
            return state
        time.sleep(.1)
    jobs.cancel(path)
    pytest.fail('Job timed out: '+jobs.log_tail(path))


def script(tmp_path, name, body):
    p = tmp_path/name
    p.write_text(f'#!{sys.executable}\n'+body)
    p.chmod(0o755)
    return str(p)


def test_index_mapping_skips_and_alternatives(spectra,tmp_path):
    manifest = identity.export_mgf(spectra,tmp_path/'representatives.mgf')
    path = tmp_path/'pred.mztab'
    write_mztab(path,[('PEPTIDEK',.7,'index=1'),('OTHER',.3,'index=1'),('PEPTIDER',.8,'scan=3')])
    dn = psm.read_casanovo_mztab(path,manifest=manifest)
    assert len(dn)==2
    row = dn[dn.scan_number==99].iloc[0]
    assert row.run_id=='run B' and row.denovo_candidate_count==2
    assert row.denovo_score_gap==pytest.approx(.4)
    assert dn.scan_number.max()==100  # not index 2 or export scan 3
    with pytest.raises(ValueError,match='manifest'):
        psm.read_casanovo_mztab(path)


def test_bad_prediction_id_is_not_guessed(spectra,tmp_path):
    manifest=identity.export_mgf(spectra,tmp_path/'representatives.mgf')
    path=tmp_path/'pred.mztab'; write_mztab(path,[('PEPTIDEK',.9,'index=999')])
    with pytest.raises(ValueError,match='resolve'):
        psm.read_casanovo_mztab(path,manifest=manifest)


def test_attach_rejects_unknown_or_duplicate_keys(spectra):
    annotation=pd.DataFrame({'run_id':['other'],'scan_number':[99],'x':[1]})
    with pytest.raises(ValueError,match='absent'): identity.attach(spectra,annotation)
    annotation=pd.concat([spectra[identity.KEYS]]*2)
    with pytest.raises(ValueError,match='Duplicate'): identity.attach(spectra,annotation)


def test_confidence_join_is_explicit(spectra):
    p=spectra[identity.KEYS].copy()
    p['peptide']=['PEPTIDEK']*3
    p['is_decoy']=[False,False,False]
    p['search_score']=[10.,20.,30.]
    p['psm_qvalue']=[.005,.2,np.nan]
    assert psm.join_psms(spectra,p).assigned.tolist()==[True,False,False]
    assert psm.join_psms(spectra,p,assume_prefiltered=True).assigned.tolist()==[True,False,True]
    p['run_id']='wrong'
    with pytest.raises(ValueError,match='run IDs'): psm.join_psms(spectra,p)
    with pytest.raises(ValueError,match='exactly one'): psm.join_psms(spectra,p,match_on_run=False)


def test_decoys_do_not_become_targets(spectra):
    p=spectra[identity.KEYS].copy(); p['peptide']='PEPTIDEK'; p['is_decoy']=True; p['psm_qvalue']=.001
    assert not psm.join_psms(spectra,p).assigned.any()


def test_charge_scaled_losses():
    mz=np.array([500.-features.H2O/2,500.-features.H2O])
    f=features.neutral_loss_features(mz,np.array([9.,1.]),500.,charge=2)
    assert f['loss_h2o_frac']==pytest.approx(.9)


def test_complementary_ions_cover_same_bond():
    seq='PEPTIDEK'
    ions=fragments.theoretical_fragments(seq,1)
    mz=np.array([x['mz'] for x in ions if x['label'] in ('b2','y6')])
    ann=annotate_spectrum(mz,np.ones(len(mz)),seq,2)
    assert ann['bond_coverage']==pytest.approx(1/7)


def test_modified_fragment_check_uses_exact_masses(spectra):
    seq='AC[+57.021464]DEK'
    ions=fragments.theoretical_fragments(seq,1)
    row=spectra.iloc[[0]].copy(); row['assigned']=True; row['peptide']='ACDEK'; row['modified_peptide']=seq
    row.at[row.index[0],'_mz']=np.array([x['mz'] for x in ions]); row.at[row.index[0],'_intensity']=np.ones(len(ions))
    out=rescue.validate_psms(row)
    assert out.iloc[0].bond_coverage==pytest.approx(1.)


def test_fragpipe_integer_mass_is_not_delta():
    seq,error=fragpipe_sequence('ACDEK','AC[160]DEK','2C(57.021464)')
    assert not error and seq=='AC[+57.021464]DEK'
    _,error=fragpipe_sequence('ACDEK','AC[160]DEK','')
    assert 'Exact modification' in error


def test_unsupported_modification_fails_visibly():
    with pytest.raises(ValueError,match='Unsupported'): fragments.parse_peptide('AC[UNIMOD:4]DEK')
    residues,deltas,nt,ct=fragments.parse_peptide('[+42.0106]-AC[+57.021464]DEK')
    assert nt==pytest.approx(42.0106) and deltas[1]==pytest.approx(57.021464)


def test_raw_denovo_confidence_is_not_a_training_label(spectra):
    spectra['denovo_score']=.999
    spectra['longest_tag']=0; spectra['n_complementary']=0
    assert (score.build_labels(spectra)==0).all()


def test_keep_polymer_option_is_effective(spectra):
    spectra['is_polymer_like']=True; spectra['longest_tag']=5
    assert not rescue.triage(spectra).is_rescue_candidate.any()
    assert rescue.triage(spectra,drop_polymers=False).is_rescue_candidate.all()


def test_builtin_clusters_cross_run_and_keeps_singleton(spectra):
    members=clustering.cluster(spectra)
    assert members.cluster_id.nunique()==2
    assert members.cluster_size.max()==2 and members.cluster_n_runs.max()==2
    assert len(clustering.representatives(spectra,members))==2


def test_imputed_charge_remains_singleton(spectra):
    spectra['charge_imputed']=True
    assert (clustering.cluster(spectra).cluster_size==1).all()


def test_peak_matching_is_one_to_one():
    a=(np.array([100.,100.005]),np.ones(2)/np.sqrt(2))
    b=(np.array([100.]),np.ones(1))
    value,count=clustering.similarity(a,b,.02)
    assert count==1 and value==pytest.approx(1/np.sqrt(2))


def test_worker_builtin_end_to_end(spectra,tmp_path):
    job=jobs.submit(spectra,spectra,{'clustering':'builtin'},tmp_path/'jobs with spaces')
    state=wait_job(job)
    assert state['state']=='complete', jobs.log_tail(job)
    df=pd.read_parquet(Path(job)/'result.parquet')
    assert len(df)==len(spectra) and df.cluster_id.nunique()==2
    assert (Path(job)/'results.zip').stat().st_size>0
    assert (Path(job)/'report.html').exists()


def test_casanovo_adapter_and_cluster_lineage(spectra,tmp_path):
    exe=script(tmp_path,'fake casanovo', '''import sys
from pathlib import Path
if '--help' in sys.argv:
 print('--output_dir --output_root --config --model'); sys.exit()
out=Path(sys.argv[sys.argv.index('--output_dir')+1])/ (sys.argv[sys.argv.index('--output_root')+1]+'.mztab')
out.write_text('MTD\\tms_run[1]-location\\tfile:///data/representatives.mgf\\nPSH\\tsequence\\tsearch_engine_score[1]\\tspectra_ref\\nPSM\\tPEPTIDEK\\t0.95\\tms_run[1]:index=0\\n')
''')
    job=jobs.submit(spectra,spectra,{'clustering':'builtin','denovo':True,'casanovo_executable':exe},tmp_path/'jobs')
    assert wait_job(job)['state']=='complete',jobs.log_tail(job)
    out=pd.read_parquet(Path(job)/'result.parquet')
    assert len(out)==3 and out.denovo_peptide.notna().sum()==2
    assert set(out.dropna(subset=['denovo_peptide']).denovo_source)=={'direct_prediction','cluster_hypothesis'}
    assert not out.assigned.any()


def test_falcon_modern_cli_and_singleton_retention(spectra,tmp_path):
    exe=script(tmp_path,'fake_falcon','''import sys
from pathlib import Path
if '--help' in sys.argv:
 print('--distance_threshold --min_matched_peaks'); sys.exit()
assert '--distance_threshold' in sys.argv and '--eps' not in sys.argv
Path(sys.argv[2]+'.csv').write_text('# falcon output\\nspectrum_id,cluster\\nrun A.99,0\\nrun B.99,0\\n')
''')
    job=jobs.submit(spectra,spectra,{'clustering':'falcon','falcon_executable':exe},tmp_path/'jobs')
    assert wait_job(job)['state']=='complete',jobs.log_tail(job)
    out=pd.read_parquet(Path(job)/'result.parquet')
    assert len(out)==3 and out.cluster_size.max()==2 and out.cluster_size.min()==1


def test_external_failure_has_no_success_result(spectra,tmp_path):
    exe=script(tmp_path,'bad_tool',"import sys\nprint('tool failure',flush=True)\nsys.exit(7)\n")
    job=jobs.submit(spectra,spectra,{'clustering':'none','denovo':True,'casanovo_executable':exe},tmp_path/'jobs')
    assert wait_job(job)['state']=='failed'
    assert not (Path(job)/'result.parquet').exists()


def test_cancel_external_process(spectra,tmp_path):
    exe=script(tmp_path,'slow_tool','''import time
print('waiting',flush=True)
time.sleep(60)
''')
    job=jobs.submit(spectra,spectra,{'clustering':'none','denovo':True,'casanovo_executable':exe},tmp_path/'jobs')
    time.sleep(.6)
    jobs.cancel(job)
    assert wait_job(job,10)['state']=='cancelled'


def test_fragpipe_adapter_retains_original_assignment(spectra,tmp_path):
    exe=script(tmp_path,'fragpipe_stub','''import sys
from pathlib import Path
args=sys.argv
assert '--headless' in args
out=Path(args[args.index('--workdir')+1])
wf=Path(args[args.index('--workflow')+1]).read_text()
assert 'database.db-path=' in wf
out.joinpath('psm.tsv').write_text('Spectrum\\tPeptide\\tProtein\\tHyperscore\\tQ-value\\nrun A.99.99.2\\tPEPTIDEK\\tP1\\t40\\t0.001\\n')
''')
    mz=tmp_path/'run A.mzML'; mz.write_text('<mzML/>')
    wf=tmp_path/'open.workflow'; wf.write_text('msfragger.run-msfragger=true\n')
    fa=tmp_path/'db.fasta'; fa.write_text('>P1\nPEPTIDEK\n')
    cfg={'clustering':'none','fragpipe':True,'fragpipe_executable':exe,'workflow':str(wf),'fasta':str(fa),'mzml_paths':[str(mz)]}
    job=jobs.submit(spectra,spectra,cfg,tmp_path/'jobs')
    assert wait_job(job)['state']=='complete',jobs.log_tail(job)
    out=pd.read_parquet(Path(job)/'result.parquet')
    assert not out.assigned.any() and out.rescue_search_assigned.sum()==1


def test_missing_tool_blocks_before_job(spectra,tmp_path):
    with pytest.raises(ValueError,match='not found'):
        jobs.submit(spectra,spectra,{'denovo':True,'casanovo_executable':'no_such_executable_239482'},tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_streamlit_renders_and_loads_background_results(spectra,tmp_path):
    from streamlit.testing.v1 import AppTest
    app=Path(__file__).resolve().parents[1]/'app.py'
    at=AppTest.from_file(str(app),default_timeout=30)
    at.session_state['qc']=spectra
    at.session_state['qc_key']='fixture'
    at.run()
    assert not at.exception, [(e.message) for e in at.exception]
    assert any(tab.label=='Run rescue' for tab in at.tabs)
    at.button(key='rescue_start').click().run()
    assert not at.exception, [(e.message) for e in at.exception]
    job=at.session_state['rescue_job']
    assert wait_job(job)['state']=='complete',jobs.log_tail(job)
    at.run()
    at.button(key='rescue_load_results').click().run()
    assert not at.exception, [(e.message) for e in at.exception]
    assert at.session_state['qc'].cluster_size.max()==2
    at.run()
    assert at.session_state['qc'].cluster_size.max()==2
    assert not at.exception


def test_external_timeout_is_failure(spectra,tmp_path):
    exe=script(tmp_path,'timeout_tool','import time\ntime.sleep(60)\n')
    cfg={'clustering':'none','denovo':True,'casanovo_executable':exe,'timeout_hours':.0001}
    job=jobs.submit(spectra,spectra,cfg,tmp_path/'jobs')
    state=wait_job(job,10)
    assert state['state']=='failed' and 'time limit' in state['message']


def test_all_unannotatable_psms_keep_checklist_schema(spectra):
    spectra['assigned']=True; spectra['peptide']='ACDEK'
    spectra['annotation_error']='Exact modification masses missing'
    out=rescue.validate_psms(spectra)
    assert (out.verdict=='warn').all() and out.bond_coverage.isna().all()


def test_loaded_file_does_not_overwrite_rescue_on_rerun(spectra,tmp_path):
    from streamlit.testing.v1 import AppTest
    path=tmp_path/'qc.parquet'; spectra.to_parquet(path)
    at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=30).run()
    at.radio(key='radio_130').set_value('Existing qc_triaged.parquet').run()
    at.text_input(key='text_input_185').set_value(str(path)).run()
    updated=at.session_state['qc'].copy(); updated['rescue_status']='sequence_hypothesis'
    at.session_state['qc']=updated
    at.run()
    assert not at.exception
    assert (at.session_state['qc'].rescue_status=='sequence_hypothesis').all()
