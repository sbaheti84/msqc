"""
Build a single self-contained HTML file: no server, no CDN, no build step.
Open it with a browser or drop it in a bucket.

For datasets past a few thousand spectra the report carries a sampled subset
of peak arrays. The full table stays in Parquet, which is what you query with
DuckDB. This file is for looking at spectra, not for analysis at scale.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .fragments import theoretical_fragments

TABLE_COLUMNS = [
    "run_id", "scan_number", "rt_min", "precursor_mz", "charge",
    "qc_score", "triage_class", "longest_tag", "n_complementary",
    "n_peaks_above_noise", "isolation_purity", "entropy",
    "peptide", "search_score", "denovo_peptide", "denovo_score",
    "cluster_id", "cluster_size", "ladder_label", "ladder_length",
    "rescue_status", "denovo_source", "rescue_bond_coverage",
    "rescue_explained_tic_frac", "rescue_precursor_error_ppm",
    "rescue_search_peptide", "rescue_search_assignment_status",
]

FEATURE_PANEL = [
    ("longest_tag", "Longest sequence tag"),
    ("n_tags_ge3", "Tags of length 3+"),
    ("n_complementary", "Complementary b/y pairs"),
    ("complementary_tic_frac", "Intensity in complement pairs"),
    ("isotope_tic_frac", "Intensity in isotope clusters"),
    ("n_peaks_above_noise", "Peaks above noise"),
    ("snr_proxy", "Base peak / noise floor"),
    ("norm_entropy", "Normalised entropy"),
    ("frac_above_precursor", "Intensity above precursor m/z"),
    ("isolation_purity", "Isolation window purity"),
    ("n_cofragmented", "Co-isolated species"),
    ("injection_time_ms", "Injection time (ms)"),
    ("ladder_length", "Repeat-ladder length"),
]

CLASS_ORDER = [
    ("identified", "Identified by search"),
    ("rescue_candidate", "Rescue candidate"),
    ("structured_unresolved", "Structured, unresolved"),
    ("polymer_contaminant", "Polymer or detergent"),
    ("low_quality_unassigned", "Too poor to interpret"),
]


def _clean(v):
    """Coerce a cell to something JSON-safe. Numbers must stay numbers or the
    table sorts lexicographically and scan 10 lands before scan 9."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if (np.isnan(v) or np.isinf(v)) else round(float(v), 5)
    if isinstance(v, str):
        return v if v.strip() else None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def _present(v) -> bool:
    """True when a peptide field actually holds a sequence. NaN is truthy in
    Python, which silently mislabels every unassigned spectrum as assigned."""
    if v is None:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    return bool(str(v).strip()) and str(v).lower() != "nan"


def _select_for_report(df: pd.DataFrame, max_spectra: int, seed: int = 0):
    """
    Keep every rescue candidate up to the cap, then fill the remaining budget
    with a stratified sample of the other classes so the scatter plot still
    shows the shape of the run.
    """
    rng = np.random.default_rng(seed)
    rescue = df[df["triage_class"] == "rescue_candidate"]
    if len(rescue) > max_spectra:
        rescue = rescue.nlargest(max_spectra, "qc_score")

    budget = max_spectra - len(rescue)
    others = df[df["triage_class"] != "rescue_candidate"]
    picks = [rescue]
    if budget > 0 and len(others):
        per_class = max(1, budget // max(others["triage_class"].nunique(), 1))
        for _, grp in others.groupby("triage_class"):
            n = min(per_class, len(grp))
            picks.append(grp.iloc[rng.choice(len(grp), n, replace=False)])
    out = pd.concat(picks).sort_values(["run_id", "scan_number"])
    return out


def build_payload(df: pd.DataFrame, max_spectra: int = 1500,
                  peak_cap: int = 120) -> dict:
    counts = df["triage_class"].value_counts().to_dict()
    total = len(df)

    sub = _select_for_report(df, max_spectra)
    have_peaks = "_mz" in sub.columns

    spectra = []
    for _, r in sub.iterrows():
        rec = {c: _clean(r.get(c)) for c in TABLE_COLUMNS if c in sub.columns}
        rec["features"] = {k: _clean(r.get(k)) for k, _ in FEATURE_PANEL
                           if k in sub.columns}
        if have_peaks and isinstance(r.get("_mz"), (list, np.ndarray)):
            mzs = list(r["_mz"])[:peak_cap]
            ints = list(r["_intensity"])[:peak_cap]
            rec["peaks"] = [[round(float(m), 4), round(float(i), 2)]
                            for m, i in zip(mzs, ints)]
        else:
            rec["peaks"] = []

        has_search = _present(r.get("peptide")) and not _present(r.get("annotation_error"))
        has_denovo = _present(r.get("denovo_peptide")) and not _present(r.get("denovo_annotation_error"))
        pep = (r.get("modified_peptide") if _present(r.get("modified_peptide"))
               else r.get("peptide")) if has_search else (
            r.get("denovo_peptide") if has_denovo else None)
        rec["fragment_source"] = ("search" if has_search else
                                  ("de novo" if has_denovo else None))
        rec["theoretical"] = []
        if _present(pep):
            try:
                fragments = theoretical_fragments(pep)
            except ValueError as exc:
                fragments = []
                rec["annotation_error"] = str(exc)
                rec["fragment_source"] = None
            for t in fragments:
                rec["theoretical"].append(
                    [round(t["mz"], 4), t["label"], t["series"]])
        spectra.append(rec)

    return {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "total_ms2": int(total),
        "counts": {k: int(v) for k, v in counts.items()},
        "runs": sorted(df["run_id"].astype(str).unique().tolist()),
        "scorer": str(df["qc_scorer"].iloc[0]) if "qc_scorer" in df else "rule",
        "shown": len(spectra),
        "has_peaks": bool(have_peaks),
        "feature_labels": dict(FEATURE_PANEL),
        "class_labels": dict(CLASS_ORDER),
        "spectra": spectra,
    }


def write_report(df: pd.DataFrame, path: str, title: str = "Spectrum triage",
                 max_spectra: int = 1500, peak_cap: int = 120) -> str:
    payload = build_payload(df, max_spectra, peak_cap)
    html = TEMPLATE.replace("__TITLE__", title)
    html = html.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    with open(path, "w") as fh:
        fh.write(html)
    return path


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --paper:#F6F7F5; --panel:#FFFFFF; --ink:#14181B; --muted:#6E7877;
  --rule:#DBDFDB; --rule-strong:#B9BFBB;
  --identified:#2E5E86; --rescue:#B24A1E; --structured:#5E7355;
  --polymer:#857A2E; --lowq:#BCC2BD;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --mono: "SF Mono", "JetBrains Mono", "Menlo", Consolas, monospace;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:var(--paper); color:var(--ink); font-family:var(--sans);
  font-size:14px; line-height:1.5; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1560px;margin:0 auto;padding:0 24px 64px}

header{
  display:flex;align-items:baseline;gap:20px;flex-wrap:wrap;
  padding:26px 0 18px;border-bottom:1px solid var(--rule-strong);
}
header h1{font-size:19px;font-weight:600;letter-spacing:-.015em;margin:0}
header .meta{color:var(--muted);font-size:13px}
header .meta b{font-family:var(--mono);font-weight:500;color:var(--ink)}

/* fate bar: the hero. every MS2 scan in the dataset, by outcome. */
.fate{margin:26px 0 8px}
.fate h2{font-size:13px;font-weight:600;margin:0 0 10px;color:var(--muted)}
.bar{display:flex;height:46px;width:100%;overflow:hidden;
     border:1px solid var(--rule-strong)}
.bar span{display:block;cursor:pointer;transition:opacity .12s}
.bar span:hover{opacity:.72}
.bar span.dim{opacity:.28}
.legend{display:flex;flex-wrap:wrap;gap:2px 0;margin-top:12px}
.legend button{
  all:unset;cursor:pointer;display:flex;align-items:baseline;gap:8px;
  padding:7px 16px 7px 12px;border-right:1px solid var(--rule);
}
.legend button:last-child{border-right:none}
.legend button:focus-visible{outline:2px solid var(--ink);outline-offset:-2px}
.legend .swatch{width:9px;height:9px;flex:none;transform:translateY(-1px)}
.legend .n{font-family:var(--mono);font-size:14px}
.legend .lbl{font-size:12.5px;color:var(--muted)}
.legend button.off .n,.legend button.off .lbl{color:var(--rule-strong)}
.legend button.off .swatch{opacity:.3}

.cols{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.05fr);
      gap:28px;margin-top:30px;align-items:start}
@media (max-width:1080px){.cols{grid-template-columns:1fr}}

.panel{background:var(--panel);border:1px solid var(--rule)}
.panel h3{font-size:12.5px;font-weight:600;margin:0;padding:11px 14px;
          border-bottom:1px solid var(--rule);color:var(--muted)}
.panel .body{padding:14px}

.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;
          padding:10px 14px;border-bottom:1px solid var(--rule);font-size:12.5px}
.controls label{color:var(--muted);display:flex;align-items:center;gap:7px}
.controls input[type=range]{width:112px;accent-color:var(--ink)}
.controls input[type=search]{
  font:inherit;padding:5px 8px;border:1px solid var(--rule-strong);
  background:var(--paper);width:150px}
.controls output{font-family:var(--mono);color:var(--ink);min-width:30px}

#scatter{width:100%;height:250px;display:block;cursor:crosshair}
.axis{display:flex;justify-content:space-between;font-family:var(--mono);
      font-size:11px;color:var(--muted);padding:2px 14px 10px}

.tablewrap{max-height:400px;overflow:auto}
table{width:100%;border-collapse:collapse;font-size:12.5px}
thead th{position:sticky;top:0;background:var(--panel);text-align:left;
  font-weight:600;color:var(--muted);padding:8px 10px;
  border-bottom:1px solid var(--rule-strong);cursor:pointer;white-space:nowrap}
thead th:hover{color:var(--ink)}
tbody td{padding:6px 10px;border-bottom:1px solid var(--rule);white-space:nowrap}
tbody tr{cursor:pointer}
tbody tr:hover{background:#EFF2EE}
tbody tr.sel{background:#E4EAF0;box-shadow:inset 3px 0 0 var(--ink)}
td.num{font-family:var(--mono);text-align:right}
td.pep{font-family:var(--mono);max-width:190px;overflow:hidden;
       text-overflow:ellipsis}
.tag{display:inline-block;width:8px;height:8px;margin-right:6px}

#spectrum{width:100%;height:330px;display:block;cursor:crosshair}
.readout{font-family:var(--mono);font-size:12px;color:var(--muted);
         padding:0 14px 10px;min-height:18px}
.specmeta{display:flex;gap:22px;flex-wrap:wrap;padding:12px 14px 4px;
          border-bottom:1px solid var(--rule)}
.specmeta div span{display:block;font-size:11.5px;color:var(--muted)}
.specmeta div b{font-family:var(--mono);font-weight:500;font-size:14px}

.feat{display:grid;grid-template-columns:1fr auto;gap:1px 14px;padding:12px 14px}
.feat dt{font-size:12.5px;color:var(--muted);padding:3px 0}
.feat dd{margin:0;font-family:var(--mono);font-size:12.5px;padding:3px 0;
         text-align:right}
.feat dd.warn{color:var(--rescue)}

.empty{padding:40px 14px;color:var(--muted);font-size:13px;text-align:center}
.note{margin-top:26px;padding:14px 16px;border-left:3px solid var(--rescue);
      background:var(--panel);font-size:12.5px;color:var(--muted);max-width:760px}
.note b{color:var(--ink);font-weight:600}
kbd{font-family:var(--mono);font-size:11px;border:1px solid var(--rule-strong);
    padding:1px 4px;background:var(--paper)}
</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>__TITLE__</h1>
  <div class="meta">
    <b id="hTotal"></b> MS2 scans &nbsp;·&nbsp; <span id="hRuns"></span>
    &nbsp;·&nbsp; scored by <b id="hScorer"></b> &nbsp;·&nbsp; <span id="hGen"></span>
  </div>
</header>

<section class="fate">
  <h2>What happened to every MS2 scan</h2>
  <div class="bar" id="fateBar"></div>
  <div class="legend" id="legend"></div>
</section>

<div class="cols">
  <div>
    <div class="panel">
      <h3>Quality against retention time</h3>
      <div class="controls">
        <label>Minimum quality
          <input type="range" id="qcMin" min="0" max="1" step="0.02" value="0">
          <output id="qcMinOut">0.00</output>
        </label>
        <label>Minimum tag
          <input type="range" id="tagMin" min="0" max="8" step="1" value="0">
          <output id="tagMinOut">0</output>
        </label>
        <input type="search" id="search" placeholder="peptide or scan">
      </div>
      <canvas id="scatter"></canvas>
      <div class="axis"><span>retention time &rarr;</span>
        <span id="scatterCount"></span></div>
    </div>

    <div class="panel" style="margin-top:22px">
      <h3>Spectra <span id="tableCount" style="font-weight:400"></span></h3>
      <div class="tablewrap">
        <table id="tbl">
          <thead><tr id="thead"></tr></thead>
          <tbody id="tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <div>
    <div class="panel" id="specPanel">
      <h3>Spectrum <span id="specTitle" style="font-weight:400"></span></h3>
      <div class="specmeta" id="specMeta"></div>
      <canvas id="spectrum"></canvas>
      <div class="readout" id="readout"></div>
    </div>

    <div class="panel" style="margin-top:22px">
      <h3>Why this spectrum was flagged</h3>
      <dl class="feat" id="featList"></dl>
    </div>
  </div>
</div>

<div class="note">
  <b>Read this before believing anything below.</b> A rescue candidate is a
  spectrum that looks structurally strong and was not explained by the search.
  That is a hypothesis, not an identification. Weight your confidence by how
  many spectra cluster together and how many samples they came from, not by the
  quality score. A single high-scoring de novo hit is noise until it repeats.
</div>

</div>
<script>
const DATA = __DATA__;
const CLASS_COLOUR = {
  identified:'var(--identified)', rescue_candidate:'var(--rescue)',
  structured_unresolved:'var(--structured)',
  polymer_contaminant:'var(--polymer)', low_quality_unassigned:'var(--lowq)'
};
const CLASS_HEX = {
  identified:'#2E5E86', rescue_candidate:'#B24A1E',
  structured_unresolved:'#5E7355', polymer_contaminant:'#857A2E',
  low_quality_unassigned:'#BCC2BD'
};
const ORDER = ['identified','rescue_candidate','structured_unresolved',
               'polymer_contaminant','low_quality_unassigned'];

const state = {
  hidden:new Set(), qcMin:0, tagMin:0, query:'',
  sortKey:'qc_score', sortDir:-1, selected:null
};

const $ = id => document.getElementById(id);
const fmt = (v,d=2) => (v===null||v===undefined) ? '—' :
  (typeof v==='number' ? (Number.isInteger(v)?v:v.toFixed(d)) : v);

/* ---------- header + fate bar ---------- */
$('hTotal').textContent = DATA.total_ms2.toLocaleString();
$('hRuns').textContent = DATA.runs.length===1 ? DATA.runs[0]
  : DATA.runs.length + ' runs';
$('hScorer').textContent = DATA.scorer;
$('hGen').textContent = DATA.generated;

function drawFate(){
  const bar = $('fateBar'), leg = $('legend');
  bar.innerHTML=''; leg.innerHTML='';
  const total = DATA.total_ms2 || 1;
  ORDER.forEach(k=>{
    const n = DATA.counts[k]||0;
    if(n>0){
      const s=document.createElement('span');
      s.style.width=(n/total*100)+'%';
      s.style.background=CLASS_HEX[k];
      s.title=DATA.class_labels[k]+': '+n.toLocaleString();
      if(state.hidden.has(k)) s.classList.add('dim');
      s.onclick=()=>toggle(k);
      bar.appendChild(s);
    }
    const b=document.createElement('button');
    b.className = state.hidden.has(k)?'off':'';
    b.innerHTML = `<i class="swatch" style="background:${CLASS_HEX[k]}"></i>`+
      `<span class="n">${(n||0).toLocaleString()}</span>`+
      `<span class="lbl">${DATA.class_labels[k]||k}</span>`;
    b.onclick=()=>toggle(k);
    leg.appendChild(b);
  });
}
function toggle(k){
  state.hidden.has(k)?state.hidden.delete(k):state.hidden.add(k);
  drawFate(); refresh();
}

/* ---------- filtering ---------- */
function filtered(){
  const q = state.query.toLowerCase();
  return DATA.spectra.filter(s=>{
    if(state.hidden.has(s.triage_class)) return false;
    if((s.qc_score??0) < state.qcMin) return false;
    if((s.longest_tag??0) < state.tagMin) return false;
    if(q){
      const hay = (s.peptide||'')+' '+(s.denovo_peptide||'')+' '+
                  s.scan_number+' '+(s.run_id||'');
      if(!hay.toLowerCase().includes(q)) return false;
    }
    return true;
  });
}

/* ---------- scatter ---------- */
const sc = $('scatter'), sctx = sc.getContext('2d');
let scPoints=[];
function drawScatter(rows){
  const dpr = window.devicePixelRatio||1;
  const w = sc.clientWidth, h = 250;
  sc.width=w*dpr; sc.height=h*dpr; sc.style.height=h+'px';
  sctx.setTransform(dpr,0,0,dpr,0,0);
  sctx.clearRect(0,0,w,h);
  const pad={l:38,r:12,t:12,b:22};
  const rts = rows.map(r=>r.rt_min??0);
  const minX = rows.length?Math.min(...rts):0, maxX = rows.length?Math.max(...rts):1;
  const spanX = (maxX-minX)||1;

  sctx.strokeStyle='#DBDFDB'; sctx.lineWidth=1;
  sctx.fillStyle='#6E7877'; sctx.font='10px var(--mono), monospace';
  [0,0.25,0.5,0.75,1].forEach(v=>{
    const y = pad.t+(1-v)*(h-pad.t-pad.b);
    sctx.beginPath(); sctx.moveTo(pad.l,y+.5); sctx.lineTo(w-pad.r,y+.5); sctx.stroke();
    sctx.fillText(v.toFixed(2), 6, y+3);
  });

  scPoints = rows.map(r=>{
    const x = pad.l + ((r.rt_min??0)-minX)/spanX*(w-pad.l-pad.r);
    const y = pad.t + (1-(r.qc_score??0))*(h-pad.t-pad.b);
    return {x,y,r};
  });
  scPoints.forEach(p=>{
    sctx.fillStyle = CLASS_HEX[p.r.triage_class]||'#999';
    sctx.globalAlpha = p.r.triage_class==='rescue_candidate'?0.95:0.45;
    const rad = p.r.triage_class==='rescue_candidate'?3:2;
    sctx.beginPath(); sctx.arc(p.x,p.y,rad,0,6.2832); sctx.fill();
  });
  sctx.globalAlpha=1;
  if(state.selected){
    const p = scPoints.find(p=>p.r===state.selected);
    if(p){ sctx.strokeStyle='#14181B'; sctx.lineWidth=1.5;
      sctx.beginPath(); sctx.arc(p.x,p.y,6,0,6.2832); sctx.stroke(); }
  }
  $('scatterCount').textContent = rows.length.toLocaleString()+' shown';
}
sc.onclick = e=>{
  const rect = sc.getBoundingClientRect();
  const x = e.clientX-rect.left, y = e.clientY-rect.top;
  let best=null, bd=15;
  scPoints.forEach(p=>{ const d=Math.hypot(p.x-x,p.y-y); if(d<bd){bd=d;best=p;} });
  if(best) select(best.r);
};

/* ---------- table ---------- */
const COLS = [
  ['scan_number','Scan',0],['rt_min','RT',2],['precursor_mz','m/z',4],
  ['charge','z',0],['qc_score','Quality',3],['longest_tag','Tag',0],
  ['n_complementary','Compl',0],['isolation_purity','Purity',2],
  ['peptide','Search peptide',null],['denovo_peptide','De novo',null],
  ['cluster_size','Cluster',0]
];
function drawTable(rows){
  const th = $('thead'); th.innerHTML='';
  COLS.forEach(([k,label])=>{
    const el=document.createElement('th');
    el.textContent = label + (state.sortKey===k?(state.sortDir<0?' ↓':' ↑'):'');
    el.onclick=()=>{
      if(state.sortKey===k) state.sortDir*=-1;
      else {state.sortKey=k; state.sortDir=-1;}
      refresh();
    };
    th.appendChild(el);
  });

  const sorted = rows.slice().sort((a,b)=>{
    const x=a[state.sortKey], y=b[state.sortKey];
    if(x===null||x===undefined) return 1;
    if(y===null||y===undefined) return -1;
    return (x>y?1:x<y?-1:0)*state.sortDir;
  }).slice(0,600);

  const tb=$('tbody'); tb.innerHTML='';
  if(!sorted.length){
    tb.innerHTML='<tr><td colspan="11" class="empty">'+
      'No spectra match these filters. Lower the quality threshold or '+
      're-enable a category above.</td></tr>';
  }
  sorted.forEach(r=>{
    const tr=document.createElement('tr');
    if(r===state.selected) tr.className='sel';
    COLS.forEach(([k,,dec],i)=>{
      const td=document.createElement('td');
      const isNum = dec!==null;
      td.className = isNum?'num':'pep';
      let txt = isNum?fmt(r[k],dec):(r[k]||'—');
      if(i===0) td.innerHTML=`<i class="tag" style="background:${
        CLASS_HEX[r.triage_class]}"></i>${txt}`;
      else td.textContent=txt;
      tr.appendChild(td);
    });
    tr.onclick=()=>select(r);
    tb.appendChild(tr);
  });
  $('tableCount').textContent = '· showing '+sorted.length+' of '+
    rows.length.toLocaleString();
}

/* ---------- spectrum viewer ---------- */
const sp = $('spectrum'), pctx = sp.getContext('2d');
let spGeom=null;
function drawSpectrum(r){
  const dpr=window.devicePixelRatio||1;
  const w=sp.clientWidth, h=330;
  sp.width=w*dpr; sp.height=h*dpr; sp.style.height=h+'px';
  pctx.setTransform(dpr,0,0,dpr,0,0);
  pctx.clearRect(0,0,w,h);
  if(!r || !r.peaks || !r.peaks.length){
    pctx.fillStyle='#6E7877'; pctx.font='13px sans-serif';
    pctx.fillText(r? 'Peak arrays were not kept for this spectrum. Re-run '+
      'extraction with --keep-peaks.' : 'Select a spectrum.', 14, 40);
    spGeom=null; return;
  }
  const pad={l:12,r:12,t:14,b:26};
  const mzs=r.peaks.map(p=>p[0]);
  const theo=r.theoretical||[];
  const lo=Math.min(...mzs)-20, hi=Math.max(...mzs)+20;
  const span=hi-lo;
  const mid = theo.length ? pad.t+(h-pad.t-pad.b)*0.56 : h-pad.b;
  const topH = mid-pad.t, botH = h-pad.b-mid;
  const X = m => pad.l + (m-lo)/span*(w-pad.l-pad.r);
  spGeom={lo,hi,span,X,pad,w,h,mid,r};

  pctx.strokeStyle='#DBDFDB';
  pctx.beginPath(); pctx.moveTo(pad.l,mid+.5); pctx.lineTo(w-pad.r,mid+.5);
  pctx.stroke();

  // annotate observed peaks against theoretical fragments
  const ann = new Map();
  const tol=0.02;
  theo.forEach(t=>{
    let bi=-1, bd=tol;
    for(let i=0;i<r.peaks.length;i++){
      const d=Math.abs(r.peaks[i][0]-t[0]);
      if(d<bd){bd=d;bi=i;}
    }
    if(bi>=0 && !ann.has(bi)) ann.set(bi,t);
  });

  // observed, upward
  r.peaks.forEach((p,i)=>{
    const x=X(p[0]), hgt=p[1]/100*topH;
    const a=ann.get(i);
    pctx.strokeStyle = a ? (a[2]==='b'?'#2E5E86':'#B24A1E') : '#B9BFBB';
    pctx.lineWidth = a?1.4:1;
    pctx.beginPath(); pctx.moveTo(x+.5,mid); pctx.lineTo(x+.5,mid-hgt); pctx.stroke();
  });
  // labels for the strongest annotated peaks
  const labelled=[...ann.entries()]
    .sort((a,b)=>r.peaks[b[0]][1]-r.peaks[a[0]][1]).slice(0,14);
  pctx.font='10px var(--mono), monospace';
  labelled.forEach(([i,t])=>{
    const x=X(r.peaks[i][0]), hgt=r.peaks[i][1]/100*topH;
    pctx.fillStyle = t[2]==='b'?'#2E5E86':'#B24A1E';
    pctx.fillText(t[1], x-7, mid-hgt-4);
  });

  // theoretical, downward
  if(theo.length){
    const maxIdx = Math.max(...theo.map(t=>parseInt(t[1].replace(/\D/g,''))||1));
    theo.forEach(t=>{
      if(t[0]<lo||t[0]>hi) return;
      const x=X(t[0]);
      const idx=parseInt(t[1].replace(/\D/g,''))||1;
      const hgt=(0.35+0.55*(idx/maxIdx))*botH;
      pctx.strokeStyle = t[2]==='b'?'#8FB0C9':'#D9A188';
      pctx.lineWidth=1;
      pctx.beginPath(); pctx.moveTo(x+.5,mid); pctx.lineTo(x+.5,mid+hgt); pctx.stroke();
    });
  }

  // m/z axis ticks
  pctx.fillStyle='#6E7877'; pctx.font='10px var(--mono), monospace';
  for(let k=0;k<=5;k++){
    const m=lo+span*k/5;
    pctx.fillText(m.toFixed(0), X(m)-12, h-8);
  }
  const matched = ann.size, tot=r.peaks.length;
  $('readout').textContent = theo.length
    ? `${matched}/${tot} peaks matched to ${r.fragment_source} peptide `+
      `${r.peptide||r.denovo_peptide}  ·  b ions blue, y ions orange`
    : `${tot} peaks  ·  no peptide assigned, nothing to mirror against`;
}
sp.onmousemove = e=>{
  if(!spGeom) return;
  const rect=sp.getBoundingClientRect();
  const x=e.clientX-rect.left;
  const mz = spGeom.lo + (x-spGeom.pad.l)/(spGeom.w-spGeom.pad.l-spGeom.pad.r)*spGeom.span;
  const r=spGeom.r;
  let best=null,bd=1e9;
  r.peaks.forEach(p=>{const d=Math.abs(p[0]-mz); if(d<bd){bd=d;best=p;}});
  if(best && bd<3) $('readout').textContent =
    `m/z ${best[0].toFixed(4)}   relative intensity ${best[1].toFixed(1)}%`;
};
sp.onmouseleave = ()=>drawSpectrum(state.selected);

/* ---------- selection ---------- */
function select(r){
  state.selected=r;
  $('specTitle').textContent = r? `· ${r.run_id} scan ${r.scan_number}` : '';
  const meta=$('specMeta'); meta.innerHTML='';
  if(r){
    const items=[
      ['Precursor', fmt(r.precursor_mz,4)],
      ['Charge', r.charge+'+'],
      ['RT (min)', fmt(r.rt_min,2)],
      ['Quality', fmt(r.qc_score,3)],
      ['Class', DATA.class_labels[r.triage_class]||r.triage_class],
      ['Cluster size', r.cluster_size??'—']
    ];
    items.forEach(([k,v])=>{
      const d=document.createElement('div');
      d.innerHTML=`<span>${k}</span><b>${v}</b>`;
      meta.appendChild(d);
    });
  }
  const fl=$('featList'); fl.innerHTML='';
  if(r && r.features){
    Object.entries(DATA.feature_labels).forEach(([k,label])=>{
      if(!(k in r.features)) return;
      const dt=document.createElement('dt'); dt.textContent=label;
      const dd=document.createElement('dd');
      dd.textContent=fmt(r.features[k],3);
      if(k==='isolation_purity' && r.features[k]!==null && r.features[k]<0.5)
        dd.className='warn';
      if(k==='ladder_length' && (r.features[k]||0)>=5) dd.className='warn';
      fl.appendChild(dt); fl.appendChild(dd);
    });
  } else {
    fl.innerHTML='<dt style="grid-column:1/-1">Select a spectrum to see its '+
      'feature breakdown.</dt>';
  }
  drawSpectrum(r);
  refresh(true);
}

/* ---------- wiring ---------- */
function refresh(skipSelect){
  const rows=filtered();
  drawScatter(rows);
  drawTable(rows);
}
$('qcMin').oninput = e=>{state.qcMin=+e.target.value;
  $('qcMinOut').textContent=state.qcMin.toFixed(2); refresh();};
$('tagMin').oninput = e=>{state.tagMin=+e.target.value;
  $('tagMinOut').textContent=state.tagMin; refresh();};
$('search').oninput = e=>{state.query=e.target.value; refresh();};
window.addEventListener('resize',()=>{refresh(); drawSpectrum(state.selected);});
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT') return;
  const rows=filtered();
  if(!rows.length) return;
  const i=rows.indexOf(state.selected);
  if(e.key==='j'||e.key==='ArrowDown'){e.preventDefault();
    select(rows[Math.min(i+1,rows.length-1)]||rows[0]);}
  if(e.key==='k'||e.key==='ArrowUp'){e.preventDefault();
    select(rows[Math.max(i-1,0)]||rows[0]);}
});

drawFate();
refresh();
const firstRescue = DATA.spectra.find(s=>s.triage_class==='rescue_candidate');
select(firstRescue || DATA.spectra[0] || null);
</script>
</body>
</html>
"""
