"""Streamlit controls for background rescue jobs."""
from pathlib import Path
import json
import hashlib
import os
import uuid
import pandas as pd
import streamlit as st
from . import jobs


def uploaded_path(upload, category):
    if upload is None:
        return None
    data = upload.getvalue()
    name = Path(upload.name).name
    directory = Path(st.session_state['workdir'])/'rescue_inputs'
    directory.mkdir(exist_ok=True)
    path = directory/(category+'_'+hashlib.sha256(data).hexdigest()[:16]+'_'+name)
    if not path.exists():
        path.write_bytes(data)
    return str(path)


@st.fragment(run_every=2)
def job_panel(job):
    state = jobs.status(job)
    st.progress(float(state.get('progress',0)), text=state.get('message',''))
    st.caption(f"Job {Path(job).name} · {state['state']}")
    active = state['state'] not in jobs.TERMINAL
    if active and st.button('Cancel job', key='rescue_cancel'):
        jobs.cancel(job)
        st.info('Cancellation requested. Waiting for the current tool to stop.')
    with st.expander('Execution log', expanded=state['state'] == 'failed'):
        st.code(jobs.log_tail(job), language='text')
    result = Path(job)/('result.parquet' if state['state']=='complete' else 'partial.parquet')
    if not active and result.exists():
        label = 'Load rescue results into dashboard' if state['state']=='complete' else 'Load completed stages into dashboard'
        if st.button(label, key='rescue_load_results', type='primary'):
            st.session_state['qc'] = pd.read_parquet(result)
            st.session_state['qc_key'] = str(uuid.uuid4())
            st.rerun()
    if state['state'] == 'complete':
        summary = Path(job)/'summary.json'
        if summary.exists():
            st.json(json.loads(summary.read_text()))
        # One selected artifact limits browser memory use for large result sets.
        options = ['results.zip','result.parquet','rescue_evidence.csv','report.html','worker.log']
        selected = st.selectbox('Download artifact', options, key='rescue_download_choice')
        path = Path(job)/selected
        with path.open('rb') as fh:
            st.download_button('Download selected artifact',fh,file_name=selected,key='rescue_download')
    elif state['state'] in ('failed','cancelled'):
        st.warning(state['message'])
        st.caption('Completed stages remain in this job folder. Start a new job to retry with revised settings.')
    st.caption(f'Files saved on the Streamlit host: {job}')


def page_rescue(df, view):
    st.subheader('Run rescue')
    st.caption('Run clustering, de novo sequencing, and a FragPipe search here. Each job keeps its input snapshot, settings, logs, spectrum mapping, and results.')
    current = st.session_state.get('rescue_job')
    active = any(jobs.status(p)['state'] not in jobs.TERMINAL for p in st.session_state.get('rescue_history', []))
    if current:
        job_panel(current)
    st.divider()
    scope = st.radio('Clustering and de novo queue', ['All rescue candidates','Candidates in current sidebar filters'],horizontal=True,key='rescue_scope')
    source = df if scope == 'All rescue candidates' else view
    candidates = source[source['is_rescue_candidate'].fillna(False)].copy()
    st.metric('Spectra in selected queue',f'{len(candidates):,}')
    if 'charge_imputed' in candidates and candidates.charge_imputed.fillna(False).any():
        st.warning('Some precursor charges were assumed during extraction. Built-in clustering keeps those spectra as singletons; their de novo predictions still depend on that assumed charge.')
    c = st.columns(3)
    backend = c[0].selectbox('Clustering', ['builtin','falcon','none'],
        format_func=lambda x:{'builtin':'Built-in (up to 50,000 spectra)','falcon':'Falcon','none':'Skip clustering'}[x],key='rescue_backend')
    denovo = c[1].checkbox('Run Casanovo de novo',key='rescue_denovo')
    search = c[2].checkbox('Run FragPipe workflow',key='rescue_search')
    with st.expander('Clustering settings',expanded=True):
        c = st.columns(4)
        ppm = c[0].number_input('Precursor tolerance (ppm)',min_value=1.,max_value=500.,value=20.,key='rescue_ppm')
        da = c[1].number_input('Fragment tolerance (Da)',min_value=.001,max_value=1.,value=.02,format='%.3f',key='rescue_da')
        cosine = c[2].slider('Minimum cosine similarity',.5,1.,.8,.01,key='rescue_cosine')
        matches = c[3].number_input('Minimum matched peaks',min_value=1,max_value=50,value=6,key='rescue_matches')
        st.caption('Built-in clustering uses precursor/charge constraints and one-to-one fragment matching to a fixed representative. Settings need validation for your instrument. A repeated cluster supports reproducibility, not peptide identity.')
    cfg = dict(clustering=backend,denovo=denovo,fragpipe=search,precursor_ppm=ppm,
               fragment_da=da,min_cosine=cosine,min_matches=int(matches))
    if backend == 'falcon':
        cfg['falcon_executable'] = st.text_input('Falcon executable',value='falcon',key='rescue_falcon_path')
    if denovo:
        with st.expander('Casanovo settings',expanded=True):
            cfg['casanovo_executable'] = st.text_input('Casanovo executable',value='casanovo',key='rescue_casanovo_path')
            cfg['casanovo_model'] = st.text_input('Model checkpoint path (optional)',key='rescue_model_path')
            cfg['casanovo_config'] = st.text_input('Casanovo YAML configuration path (optional)',key='rescue_config_path')
            config_upload = st.file_uploader('Or upload Casanovo YAML',type=['yaml','yml'],key='rescue_config_upload')
            if config_upload is not None:
                cfg['casanovo_config'] = uploaded_path(config_upload,'casanovo')
            st.caption('An executable in a separate Conda environment can be used by its full path. Without a checkpoint Casanovo may download compatible weights. CPU/GPU selection and batch size follow its YAML configuration.')
    if search:
        with st.expander('FragPipe settings',expanded=True):
            st.info('FragPipe searches the original mzML files, preserving their acquisition context and native scan IDs. Choose an Open, mass-offset, semi-tryptic, or other saved workflow. MSBooster and PTM-Shepherd run when enabled in that workflow.')
            cfg['fragpipe_executable'] = st.text_input('FragPipe executable',value='fragpipe',key='rescue_fragpipe_path')
            cfg['workflow'] = st.text_input('FragPipe .workflow path',key='rescue_workflow_path')
            cfg['fasta'] = st.text_input('FASTA path (including decoys configured by the workflow)',key='rescue_fasta_path')
            uploads = st.columns(2)
            workflow_upload = uploads[0].file_uploader('Or upload a workflow',type=['workflow'],key='rescue_workflow_upload')
            fasta_upload = uploads[1].file_uploader('Or upload a FASTA',type=['fasta','fa','faa'],key='rescue_fasta_upload')
            if workflow_upload is not None:
                cfg['workflow'] = uploaded_path(workflow_upload,'workflow')
            if fasta_upload is not None:
                cfg['fasta'] = uploaded_path(fasta_upload,'database')
            text = st.text_area('Original mzML paths, one per line',value='\n'.join(st.session_state.get('mzml_paths',[])),key='rescue_mzml_paths')
            cfg['mzml_paths'] = [p.strip() for p in text.splitlines() if p.strip()]
            c = st.columns(3)
            cfg['tools_folder'] = c[0].text_input('FragPipe tools folder (first setup)',key='rescue_tools_folder')
            cfg['diann'] = c[1].text_input('DIA-NN executable (if required)',key='rescue_diann')
            cfg['python_folder'] = c[2].text_input('FragPipe Python folder (if required)',key='rescue_python_folder')
            cfg['search_prefiltered'] = st.checkbox('The selected workflow produces FDR-filtered psm.tsv files',key='rescue_search_filtered',
                help='Required to accept results without explicit q-values. Otherwise they remain tentative. The workflow controls its own FDR settings.')
            cfg['max_qvalue'] = st.number_input('Maximum reported PSM q-value',min_value=0.,max_value=1.,value=.01,format='%.4f',key='rescue_search_qvalue')
    with st.expander('Resources and tool availability'):
        c = st.columns(3)
        cfg['threads'] = int(c[0].number_input('FragPipe CPU threads',min_value=1,max_value=256,value=4,key='rescue_threads'))
        cfg['ram_gb'] = int(c[1].number_input('FragPipe memory limit (GB)',min_value=1,max_value=1024,value=16,key='rescue_ram'))
        cfg['timeout_hours'] = c[2].number_input('Job time limit (hours)',min_value=.01,max_value=168.,value=24.,key='rescue_timeout')
        from .checklist import ANALYZER_TOLERANCE
        cfg['analyzer'] = st.selectbox('Analyzer for rescue evidence',list(ANALYZER_TOLERANCE),key='rescue_analyzer')
        st.dataframe(pd.DataFrame(jobs.availability(cfg)),hide_index=True)
        st.caption('Tools execute on the Streamlit host. Hosted services with limited RAM or no GPU may not support large external jobs. Install tools there once; then run them from this tab.')
    if st.button('Start rescue workflow',type='primary',disabled=bool(active),key='rescue_start'):
        try:
            root = Path(st.session_state['workdir'])/'jobs'
            job = jobs.submit(df,candidates,cfg,root)
            st.session_state['rescue_job'] = job
            st.session_state.setdefault('rescue_history',[]).append(job)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.rerun()
    history = st.session_state.get('rescue_history',[])
    if len(history) > 1:
        previous = st.selectbox('Job history',history,format_func=lambda p:Path(p).name,key='rescue_history_choice')
        if st.button('View selected job',key='rescue_history_open'):
            st.session_state['rescue_job'] = previous
            st.rerun()
