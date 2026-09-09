"""Background rescue worker, launched by msqc.jobs (also runnable for recovery)."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback
import zipfile
import numpy as np
import pandas as pd
from . import clustering, jobs, psm
from .identity import KEYS, attach, export_mgf


class Cancelled(Exception):
    pass


class Runner:
    def __init__(self, job):
        self.job = Path(job).resolve()
        self.config = json.loads((self.job/'config.json').read_text())
        self.deadline = time.monotonic() + float(self.config.get('timeout_hours',24))*3600
        self.progress = 0.

    def check(self):
        if (self.job/'cancel.request').exists():
            raise Cancelled('Cancelled by user; completed stages remain available.')
        if time.monotonic() > self.deadline:
            raise TimeoutError('Job time limit reached; completed stages remain available.')

    def tick(self, progress, message, state='running'):
        self.progress = progress
        jobs.write_json(self.job/'status.json', dict(state=state,progress=progress,message=message,
                                                    updated=time.time()))

    def command(self, args, label, capture=None):
        self.check()
        print(f'[{label}] {shlex.join([str(a) for a in args])}', flush=True)
        log_path = Path(capture) if capture else self.job/'worker.log'
        with log_path.open('ab') as log:
            proc = subprocess.Popen([str(a) for a in args], cwd=self.job,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=(os.name != 'nt'))
            try:
                while proc.poll() is None:
                    self.check()
                    time.sleep(.2)
                if proc.returncode:
                    raise RuntimeError(f'{label} exited with code {proc.returncode}. See worker.log and the tool logs.')
            except BaseException:
                self.stop_process(proc)
                raise
        self.check()

    @staticmethod
    def stop_process(proc):
        if os.name != 'nt':
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
        elif proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            if os.name != 'nt':
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            proc.wait()

    def help(self, executable, *subcommands):
        path = self.job / (Path(executable).stem + '_help.txt')
        self.command([executable,*subcommands,'--help'], 'Checking tool interface', capture=path)
        return path.read_text(errors='replace')

    def checkpoint(self, df, name):
        df.to_parquet(self.job/'partial.parquet', index=False)
        df.drop(columns=['_mz','_intensity'], errors='ignore').to_csv(self.job/'rescue_evidence.csv', index=False)
        print(f'Completed {name}; {len(df):,} spectrum rows retained.', flush=True)

    def fragpipe(self, df):
        cfg = self.config
        self.tick(.05, 'Running FragPipe workflow on original mzML files')
        dest = self.job/'fragpipe'
        dest.mkdir()
        workflow = Path(cfg['workflow']).read_text()
        # Java properties require escaped backslashes/colons in Windows paths.
        fasta = cfg['fasta'].replace('\\','\\\\').replace(':','\\:')
        workflow = re.sub(r'^database\.db-path=.*$', '', workflow, flags=re.M)
        workflow += '\ndatabase.db-path=' + fasta + '\n'
        path = self.job/'search.workflow'; path.write_text(workflow)
        manifest = self.job/'fragpipe.fp-manifest'
        manifest.write_text(''.join(f'{p}\tmsqc\t\tDDA\n' for p in cfg['mzml_paths']))
        args = [cfg['fragpipe_executable'],'--headless','--workflow',path,
                '--manifest',manifest,'--workdir',dest,
                '--threads',str(cfg.get('threads',4)),'--ram',str(cfg.get('ram_gb',16))]
        for key, option in [('tools_folder','--config-tools-folder'),('diann','--config-diann'),('python_folder','--config-python')]:
            if cfg.get(key):
                args.extend([option,cfg[key]])
        self.command(args, 'FragPipe')
        paths = sorted(dest.rglob('psm.tsv'))
        if not paths:
            raise RuntimeError('FragPipe produced no psm.tsv. Enable PSM reporting in the selected workflow.')
        tables = pd.concat([psm.read_fragpipe_psm(str(p)) for p in paths], ignore_index=True)
        joined = psm.join_psms(df[KEYS], tables, max_qvalue=cfg.get('max_qvalue',.01),
                              assume_prefiltered=cfg.get('search_prefiltered',False))
        joined = joined.rename(columns={c:'rescue_search_'+c for c in joined if c not in KEYS})
        df = attach(df, joined)
        self.checkpoint(df, 'FragPipe search')
        return df

    def run(self):
        cfg = self.config
        self.check()
        df = pd.read_parquet(self.job/'input.parquet')
        # A new job does not silently attribute an earlier job's predictions to
        # its selected methods. Earlier results remain in their saved job folder.
        df = df.drop(columns=[c for c in df if c.startswith(('cluster_', 'representative_', 'denovo_', 'rescue_'))])
        candidates = pd.read_parquet(self.job/'candidates.parquet')
        if cfg.get('fragpipe'):
            df = self.fragpipe(df)
        members = None
        if len(candidates):
            backend = cfg.get('clustering','builtin')
            if backend == 'builtin':
                self.tick(.25, 'Clustering rescue spectra')
                members = clustering.cluster(candidates,
                    precursor_ppm=cfg.get('precursor_ppm',20.), fragment_da=cfg.get('fragment_da',.02),
                    min_cosine=cfg.get('min_cosine',.8), min_matches=cfg.get('min_matches',6),
                    progress=lambda f,m:self.tick(.25+.25*f,m), cancelled=self.check)
            elif backend == 'falcon':
                self.tick(.25, 'Running Falcon clustering')
                exe = cfg['falcon_executable']; help_text = self.help(exe)
                distance = '--distance_threshold' if '--distance_threshold' in help_text else '--eps'
                if distance not in help_text:
                    raise RuntimeError('Unsupported Falcon CLI: no recognized distance threshold option.')
                args = [exe,self.job/'rescue_candidates.mgf',self.job/'falcon',
                        '--precursor_tol',str(cfg.get('precursor_ppm',20.)),'ppm',
                        '--fragment_tol',str(cfg.get('fragment_da',.02)),
                        distance,str(1-cfg.get('min_cosine',.8))]
                if '--min_matched_peaks' in help_text:
                    args.extend(['--min_matched_peaks',str(cfg.get('min_matches',6))])
                self.command(args,'Falcon')
                members = clustering.from_falcon(candidates,self.job/'falcon.csv')
            if members is not None:
                members.to_csv(self.job/'clusters.csv', index=False)
                df = attach(df,members)
                reps = clustering.representatives(candidates,members)
                self.checkpoint(df,'clustering')
            else:
                reps = candidates
            if cfg.get('denovo'):
                self.tick(.55, f'Sequencing {len(reps):,} representative spectra with Casanovo')
                manifest = export_mgf(reps,self.job/'representatives.mgf')
                exe = cfg['casanovo_executable']; help_text = self.help(exe,'sequence')
                if '--output_dir' in help_text and '--output_root' in help_text:
                    args = [exe,'sequence','--output_dir',self.job,'--output_root','casanovo']
                elif re.search(r'--output\b', help_text):
                    args = [exe,'sequence','--output',self.job/'casanovo.mztab']
                else:
                    raise RuntimeError('Unsupported Casanovo output options. See saved help output.')
                for key, flag in [('casanovo_config','--config'),('casanovo_model','--model')]:
                    if cfg.get(key):
                        args.extend([flag,cfg[key]])
                args.append(self.job/'representatives.mgf')
                self.command(args,'Casanovo')
                dn = psm.read_casanovo_mztab(str(self.job/'casanovo.mztab'),manifest=manifest)
                dn.to_csv(self.job/'representative_predictions.csv',index=False)
                if members is not None:
                    pred = dn.rename(columns={'run_id':'representative_run_id','scan_number':'representative_scan_number'})
                    dn = members[KEYS+['representative_run_id','representative_scan_number']].merge(
                        pred,on=['representative_run_id','representative_scan_number'],how='left',validate='many_to_one')
                    dn['denovo_source'] = np.where(
                        (dn.run_id == dn.representative_run_id) & (dn.scan_number == dn.representative_scan_number),
                        'direct_prediction','cluster_hypothesis')
                else:
                    dn['denovo_source'] = 'direct_prediction'
                df = attach(df,dn)
                df = evidence(df, cfg.get('analyzer','orbitrap_hcd'), self.check)
                self.checkpoint(df,'de novo annotation')
        self.check()
        self.tick(.93,'Writing report and download bundle')
        df['rescue_status'] = 'unresolved'
        if 'denovo_peptide' in df:
            df.loc[df.denovo_peptide.notna(),'rescue_status'] = 'sequence_hypothesis'
        if 'rescue_search_assigned' in df:
            df.loc[df.rescue_search_assigned.fillna(False),'rescue_status'] = 'search_confident'
        # Preserve the original assigned/triage fields so rescue gains can be audited.
        df.to_parquet(self.job/'result.parquet', index=False)
        df.drop(columns=['_mz','_intensity'],errors='ignore').to_csv(self.job/'rescue_evidence.csv',index=False)
        from .report import write_report
        # Signature is checked by tests alongside the normal report command.
        write_report(df, str(self.job/'report.html'), title='msqc rescue results', max_spectra=2000)
        jobs.write_json(self.job/'summary.json', {'spectra':len(df),'queued_spectra':len(candidates),
            'clusters':int(members.cluster_id.nunique()) if members is not None else None,
            'sequence_hypotheses':int(df.get('denovo_peptide',pd.Series(dtype=object)).notna().sum()),
            'new_search_matches':int((~df['assigned'].fillna(False) & df.get('rescue_search_assigned',pd.Series(False,index=df.index)).fillna(False)).sum())})
        with zipfile.ZipFile(self.job/'results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in self.job.rglob('*'):
                if p.is_file() and p.name not in {'results.zip','input.parquet','candidates.parquet','partial.parquet','cancel.request','status.json'}:
                    z.write(p,p.relative_to(self.job))
            z.writestr('status.json', json.dumps(dict(state='complete',progress=1.,message='Rescue workflow complete')))
        self.tick(1.,'Rescue workflow complete', 'complete')


def evidence(df, analyzer, check=lambda:None):
    from .checklist import ANALYZER_TOLERANCE, annotate_spectrum
    from .fragments import parse_peptide
    from .features import RESIDUE_MASSES, H2O, PROTON
    df = df.copy()
    _, ppm, da = ANALYZER_TOLERANCE[analyzer]
    for i,row in df.iterrows():
        check()
        seq = row.get('denovo_peptide')
        if not isinstance(seq,str) or not seq:
            continue
        try:
            residues,deltas,nt,ct = parse_peptide(seq)
            if row.get('denovo_modifications','null') not in ('null','0','',None) and not any(deltas) and not nt and not ct:
                raise ValueError('Modifications are reported separately; normalized modified sequence is unavailable.')
            ann = annotate_spectrum(row['_mz'],row['_intensity'],seq,int(row['charge']),frag_ppm=ppm,frag_da=da)
            for k in ('bond_coverage','longest_consecutive_series','explained_tic_frac','median_abs_error_ppm'):
                df.loc[i,'rescue_'+k] = ann[k]
            mass = sum(RESIDUE_MASSES[a]+d for a,d in zip(residues,deltas))+H2O+nt+ct
            expected = mass/int(row['charge']) + PROTON
            df.loc[i,'rescue_precursor_error_ppm'] = (row['precursor_mz']-expected)/expected*1e6
            df.loc[i,'rescue_evidence_note'] = 'Evidence only; not an FDR acceptance decision.'
        except (ValueError,TypeError) as exc:
            df.loc[i,'rescue_evidence_note'] = str(exc)
    return df


def main():
    runner = Runner(sys.argv[1])
    try:
        runner.run()
    except Cancelled as exc:
        runner.tick(runner.progress,str(exc),'cancelled')
    except Exception as exc:
        traceback.print_exc()
        runner.tick(runner.progress,str(exc),'failed')


if __name__ == '__main__':
    main()
