"""Disk-backed subprocess jobs independent of Streamlit reruns.

Each submission is an immutable input snapshot in a new directory. No shell
commands, global session cache, or user-entered command strings are evaluated.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone

TERMINAL = {'complete', 'failed', 'cancelled'}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str))
    os.replace(tmp, path)


def executable(value):
    value = str(value).strip()
    found = shutil.which(value)
    if not found:
        raise ValueError(f'Executable not found: {value}. Install it on the Streamlit host or supply its full path.')
    return str(Path(found).resolve())


def availability(config=None):
    config = config or {}
    return [{'tool': name, 'available': bool(shutil.which(config.get(key) or default)),
             'executable': config.get(key) or default}
            for name, key, default in [('Falcon','falcon_executable','falcon'),
                                       ('Casanovo','casanovo_executable','casanovo'),
                                       ('FragPipe','fragpipe_executable','fragpipe')]]


def validate_config(config):
    cfg = dict(config)
    backend = cfg.get('clustering', 'builtin')
    if backend not in ('builtin','falcon','none'):
        raise ValueError('Unknown clustering backend.')
    if not (backend != 'none' or cfg.get('denovo') or cfg.get('fragpipe')):
        raise ValueError('Select at least one rescue stage.')
    for tool, needed in [('falcon', backend == 'falcon'), ('casanovo', cfg.get('denovo')), ('fragpipe', cfg.get('fragpipe'))]:
        if needed:
            cfg[tool+'_executable'] = executable(cfg.get(tool+'_executable') or tool)
    for key in ('casanovo_model','casanovo_config','workflow','fasta'):
        if cfg.get(key):
            path = Path(cfg[key]).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f'{key}: file does not exist: {path}')
            cfg[key] = str(path)
    if cfg.get('fragpipe'):
        if not cfg.get('workflow') or not cfg.get('fasta') or not cfg.get('mzml_paths'):
            raise ValueError('FragPipe needs a workflow, FASTA, and the original mzML paths.')
        paths = [str(Path(p).expanduser().resolve()) for p in cfg['mzml_paths']]
        if any(not Path(p).is_file() or Path(p).suffix.lower() != '.mzml' for p in paths):
            raise ValueError('FragPipe input must be existing original mzML files.')
        if len({Path(p).stem for p in paths}) != len(paths):
            raise ValueError('FragPipe input run names must be unique.')
        if any(any(c in p for c in '\t\r\n') for p in paths):
            raise ValueError('Invalid characters in mzML paths.')
        cfg['mzml_paths'] = paths
    import math
    for key, default, lo, hi in [('precursor_ppm',20.,0.,500.), ('fragment_da',.02,0.,1.),
                                 ('min_cosine',.8,0.,1.), ('timeout_hours',24.,0.,168.)]:
        value = float(cfg.get(key,default))
        if not math.isfinite(value) or not lo < value <= hi:
            raise ValueError(f'Invalid {key}.')
    for key,default in [('threads',4),('ram_gb',16),('min_matches',6)]:
        if int(cfg.get(key,default)) < 1:
            raise ValueError(f'{key} must be positive.')
    if not 0 <= float(cfg.get('max_qvalue',.01)) <= 1:
        raise ValueError('Invalid maximum q-value.')
    if float(cfg.get('timeout_hours', 24)) <= 0:
        raise ValueError('Timeout must be positive.')
    return cfg


def submit(qc, candidates, config, root):
    from .identity import export_mgf
    cfg = validate_config(config)
    if candidates.empty and not cfg.get('fragpipe'):
        raise ValueError('No rescue candidates selected.')
    job = Path(root).resolve() / uuid.uuid4().hex
    job.mkdir(parents=True)
    qc.to_parquet(job/'input.parquet', index=False)
    candidates.to_parquet(job/'candidates.parquet', index=False)
    if not candidates.empty:
        export_mgf(candidates, job/'rescue_candidates.mgf')
    # Snapshot mutable configuration files, leaving large model/FASTA files in place.
    for key in ('workflow','casanovo_config'):
        if cfg.get(key):
            dest = job / ('input_' + key + Path(cfg[key]).suffix)
            shutil.copyfile(cfg[key], dest)
            cfg[key] = str(dest)
    write_json(job/'config.json', cfg)
    write_json(job/'status.json', {'state':'queued','progress':0.,'message':'Starting worker',
                                 'created':datetime.now(timezone.utc).isoformat()})
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get('PYTHONPATH','')
    with (job/'worker.log').open('ab') as log:
        try:
            proc = subprocess.Popen([sys.executable, '-m', 'msqc.worker', str(job)],
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=(os.name != 'nt'))
        except Exception as e:
            write_json(job/'status.json', {'state':'failed','progress':0.,'message':str(e)})
            raise
    write_json(job/'worker.json', {'pid':proc.pid})
    return str(job)


def status(job):
    path = Path(job)/'status.json'
    return json.loads(path.read_text())


def cancel(job):
    if status(job)['state'] not in TERMINAL:
        (Path(job)/'cancel.request').touch()


def log_tail(job, limit=16000):
    p = Path(job)/'worker.log'
    if not p.exists():
        return ''
    with p.open('rb') as fh:
        fh.seek(max(0, p.stat().st_size-limit))
        return fh.read().decode(errors='replace')
