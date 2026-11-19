#!/usr/bin/env python3
"""Drive the full Assistant-Axis run as resumable Kaggle chunks.

  build                 write full/kernels/cNN/ from chunk_template.py (+ setup cell of kernel/00_calibrate.ipynb)
  push N [N ...]        push chunk kernels (each restores finished roles from the aa-full-state dataset)
  status                Kaggle status of every chunk kernel
  collect N [N ...]     download chunk outputs into full/runs/cNN and merge finished files into full/state/
  upload-state          publish full/state/ as a new version of the private aa-full-state dataset
  summary               which roles are complete / missing per chunk
  autopilot             loop: keep one chunk running (leaves the 2nd Kaggle GPU slot free), collect + update
                        state when it ends, push the next chunk with missing roles; waits out slot/quota limits
"""
import json, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
USER = 'kritiagarwal993'
QCOUNT = 24
CHUNK_SIZE = 24
KINDS = {'responses': '.jsonl', 'scores': '.json', 'activations': '.pt'}
STATE, RUNS, KDIR = ROOT / 'state', ROOT / 'runs', ROOT / 'kernels'


def roles():
    import zipfile
    with zipfile.ZipFile(REPO / 'dataset/assistant-axis-code.zip') as z:
        names = sorted(Path(n).stem for n in z.namelist()
                       if n.startswith('data/roles/instructions/') and n.endswith('.json'))
    assert len(names) == 276, len(names)
    return names


def chunks():
    r = roles()
    return [r[i:i + CHUNK_SIZE] for i in range(0, len(r), CHUNK_SIZE)]


def kid(n): return f'{USER}/aa-full-c{n:02d}'


def sh(cmd, check=True):
    print('$', cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check, text=True)


def build():
    nb = json.load(open(REPO / 'kernel/00_calibrate.ipynb'))
    setup = ''.join(nb['cells'][4]['source'])
    tmpl = (ROOT / 'chunk_template.py').read_text()
    meta = json.load(open(REPO / 'kernel/kernel-metadata.json'))
    for n, rs in enumerate(chunks()):
        d = KDIR / f'c{n:02d}'; d.mkdir(parents=True, exist_ok=True)
        code = (tmpl.replace('__CHUNK__', str(n)).replace('__ROLES__', json.dumps(rs))
                    .replace('__QCOUNT__', str(QCOUNT)).replace('__SETUP__', setup))
        (d / 'run.py').write_text(code)
        m = dict(meta, id=kid(n), title=f'aa-full-c{n:02d}', code_file='run.py', kernel_type='script',
                 dataset_sources=[f'{USER}/assistant-axis-code', f'{USER}/aa-full-state'])
        json.dump(m, open(d / 'kernel-metadata.json', 'w'), indent=2)
    print(f'built {len(chunks())} chunks of <= {CHUNK_SIZE} roles, q{QCOUNT}')


def push(ns):
    for n in ns: sh(f'kaggle kernels push -p {KDIR / f"c{n:02d}"}')


def status():
    for n in range(len(chunks())):
        r = subprocess.run(f'kaggle kernels status {kid(n)}', shell=True, capture_output=True, text=True)
        print(f'c{n:02d}', (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr).strip() else '?')


def collect(ns):
    for n in ns:
        d = RUNS / f'c{n:02d}'
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
        sh(f'kaggle kernels output {kid(n)} -p {d}')
        for k, ext in KINDS.items():
            (STATE / k).mkdir(parents=True, exist_ok=True)
            for f in (d / 'out' / k).glob(f'*{ext}') if (d / 'out' / k).exists() else []:
                shutil.copy(f, STATE / k / f.name)
        m = d / 'manifest.json'
        print(f'c{n:02d}:', json.load(open(m))['missing'] if m.exists() else 'no manifest (run failed?)', 'missing')
    summary()


def upload_state():
    STATE.mkdir(exist_ok=True)
    meta = {'title': 'aa-full-state', 'id': f'{USER}/aa-full-state', 'licenses': [{'name': 'CC0-1.0'}]}
    json.dump(meta, open(STATE / 'dataset-metadata.json', 'w'))
    (STATE / 'README.md').write_text('Finished per-role outputs of the full Assistant-Axis run (see full/orchestrate.py).\n')
    exists = subprocess.run(f'kaggle datasets status {USER}/aa-full-state', shell=True, capture_output=True, text=True)
    if 'ready' in exists.stdout or 'pending' in exists.stdout:
        sh(f'kaggle datasets version -p {STATE} -m "state update" --dir-mode zip')
    else:
        sh(f'kaggle datasets create -p {STATE} --dir-mode zip')


def done_roles():
    return {f.stem for f in (STATE / 'activations').glob('*.pt')} if (STATE / 'activations').exists() else set()


def summary():
    done = done_roles()
    for n, rs in enumerate(chunks()):
        miss = [r for r in rs if r not in done]
        print(f'c{n:02d}: {len(rs) - len(miss):2d}/{len(rs)} done' + (f'  missing: {" ".join(miss)}' if miss and len(miss) < len(rs) else ''))
    print(f'TOTAL {len(done)}/276 roles complete')


def kstatus(n):
    r = subprocess.run(f'kaggle kernels status {kid(n)}', shell=True, capture_output=True, text=True)
    out = (r.stdout + r.stderr)
    for s in ('RUNNING', 'QUEUED', 'COMPLETE', 'ERROR', 'CANCEL'):
        if s in out: return s
    return 'NONE' if '404' in out or 'not found' in out.lower() else 'UNKNOWN'


def wait_state_ready():
    for _ in range(60):
        r = subprocess.run(f'kaggle datasets status {USER}/aa-full-state', shell=True, capture_output=True, text=True)
        if r.stdout.strip() == 'ready': return
        time.sleep(20)


def autopilot():
    stamp = lambda: time.strftime('%Y-%m-%d %H:%M')
    book = RUNS / 'autopilot.json'; RUNS.mkdir(exist_ok=True)
    st = json.load(open(book)) if book.exists() else {'current': None, 'stalls': {}}
    save = lambda: json.dump(st, open(book, 'w'), indent=1)
    while True:
        cur = st['current']
        if cur is not None:
            s = kstatus(cur)
            if s in ('RUNNING', 'QUEUED', 'UNKNOWN'):
                time.sleep(600); continue
            before = len(done_roles())
            print(f'{stamp()} c{cur:02d} ended: {s}', flush=True)
            collect([cur]); upload_state(); wait_state_ready()
            gained = len(done_roles()) - before
            st['stalls'][str(cur)] = 0 if gained else st['stalls'].get(str(cur), 0) + 1
            print(f'{stamp()} c{cur:02d} added {gained} roles; total {len(done_roles())}/276', flush=True)
            st['current'] = None; save()
        done = done_roles()
        todo = [n for n, rs in enumerate(chunks()) if any(r not in done for r in rs) and st['stalls'].get(str(n), 0) < 2]
        if not todo:
            print(f'{stamp()} AUTOPILOT FINISHED: {len(done)}/276 roles complete', flush=True); summary(); return
        n = todo[0]
        r = subprocess.run(f'kaggle kernels push -p {KDIR / f"c{n:02d}"}', shell=True, capture_output=True, text=True)
        msg = (r.stdout + r.stderr).strip()
        if 'successfully pushed' in msg:
            print(f'{stamp()} pushed c{n:02d}', flush=True); st['current'] = n; save(); time.sleep(120)
        else:
            print(f'{stamp()} push c{n:02d} refused: {msg[-200:]}', flush=True); time.sleep(900)


if __name__ == '__main__':
    cmd, args = sys.argv[1], [int(a) for a in sys.argv[2:]]
    {'build': build, 'status': status, 'summary': summary, 'upload-state': upload_state, 'autopilot': autopilot}.get(cmd, lambda: None)() \
        if cmd not in ('push', 'collect') else {'push': push, 'collect': collect}[cmd](args)
