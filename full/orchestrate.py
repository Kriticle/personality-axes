#!/usr/bin/env python3
"""Drive the full Assistant-Axis run as resumable Kaggle chunks.

  build                 write full/kernels/cNN/ from chunk_template.py (+ setup cell of kernel/00_calibrate.ipynb)
  push N [N ...]        push chunk kernels (each restores finished roles from the aa-full-state dataset)
  status                Kaggle status of every chunk kernel
  collect N [N ...]     download chunk outputs into full/runs/cNN and merge finished files into full/state/
  upload-state          publish full/state/ as a new version of the private aa-full-state dataset
  summary               which roles are complete / missing per chunk
  autopilot             loop for the account in AA_KAGGLE_USER: keep one chunk running, collect + update state when
                        it ends, claim and push the next free chunk; waits out slot/quota limits. One autopilot per
                        account runs in parallel; they share full/runs/claims.json (under a file lock).
  status-file           rewrite full/runs/STATUS.md (what is done, what is left, which account runs what)
"""
import os, json, shutil, subprocess, sys, time, fcntl, contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
# account -> access-token file (None = the CLI default ~/.kaggle/access_token). Kaggle CLI 2.x ignores
# KAGGLE_CONFIG_DIR for tokens, so every kaggle call gets KAGGLE_API_TOKEN set explicitly.
ACCOUNTS = {'kritiagarwal993': None, 'pratik1234agarwal': '~/.kaggle2/access_token'}
USER = os.environ.get('AA_KAGGLE_USER', 'kritiagarwal993')
QCOUNT = 24
CHUNK_SIZE = 24
KINDS = {'responses': '.jsonl', 'scores': '.json', 'activations': '.pt'}
STATE, RUNS, KDIR = ROOT / 'state', ROOT / 'runs', ROOT / 'kernels'
CLAIMS, STATUS_MD = RUNS / 'claims.json', RUNS / 'STATUS.md'


def kenv(user=None):
    env = dict(os.environ); env.pop('KAGGLE_API_TOKEN', None)
    tok = ACCOUNTS[user or USER]
    if tok: env['KAGGLE_API_TOKEN'] = os.path.expanduser(tok)
    return env


def kdir(n, user=None):
    user = user or USER
    return (KDIR if user == 'kritiagarwal993' else KDIR / user) / f'c{n:02d}'


@contextlib.contextmanager
def locked():
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / '.lock', 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


def load_claims():
    c = json.load(open(CLAIMS)) if CLAIMS.exists() else {}
    c.setdefault('running', {}); c.setdefault('stalls', {}); c.setdefault('accounts', {})
    return c


def save_claims(c):
    json.dump(c, open(CLAIMS, 'w'), indent=1)


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


def kid(n, user=None): return f'{user or USER}/aa-full-c{n:02d}'


def sh(cmd, check=True, user=None):
    print('$', cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check, text=True, env=kenv(user))


def kq(cmd, user=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=kenv(user))
    return (r.stdout + r.stderr).strip()


def build():
    nb = json.load(open(REPO / 'kernel/00_calibrate.ipynb'))
    setup = ''.join(nb['cells'][4]['source'])
    tmpl = (ROOT / 'chunk_template.py').read_text()
    meta = json.load(open(REPO / 'kernel/kernel-metadata.json'))
    for user in ACCOUNTS:
        for n, rs in enumerate(chunks()):
            d = kdir(n, user); d.mkdir(parents=True, exist_ok=True)
            code = (tmpl.replace('__CHUNK__', str(n)).replace('__ROLES__', json.dumps(rs))
                        .replace('__QCOUNT__', str(QCOUNT)).replace('__SETUP__', setup))
            (d / 'run.py').write_text(code)
            m = dict(meta, id=kid(n, user), title=f'aa-full-c{n:02d}', code_file='run.py', kernel_type='script',
                     dataset_sources=[f'{user}/assistant-axis-code', f'{user}/aa-full-state'])
            json.dump(m, open(d / 'kernel-metadata.json', 'w'), indent=2)
    print(f'built {len(chunks())} chunks of <= {CHUNK_SIZE} roles, q{QCOUNT}, for {", ".join(ACCOUNTS)}')


def push(ns):
    for n in ns: sh(f'kaggle kernels push -p {kdir(n)}')


def status():
    for n in range(len(chunks())):
        out = kq(f'kaggle kernels status {kid(n)}')
        print(f'c{n:02d}', out.splitlines()[-1] if out else '?')


def collect(ns):
    for n in ns:
        d = RUNS / f'c{n:02d}'
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
        sh(f'kaggle kernels output {kid(n)} -p {d}')
        (d / 'account.txt').write_text(USER)
        for k, ext in KINDS.items():
            (STATE / k).mkdir(parents=True, exist_ok=True)
            for f in (d / 'out' / k).glob(f'*{ext}') if (d / 'out' / k).exists() else []:
                shutil.copy(f, STATE / k / f.name)
        m = d / 'manifest.json'
        print(f'c{n:02d}:', json.load(open(m))['missing'] if m.exists() else 'no manifest (run failed?)', 'missing')
    summary()


def upload_state():
    """Publish full/state/ to every account's aa-full-state, so whichever account runs a chunk next sees all of it."""
    STATE.mkdir(exist_ok=True)
    (STATE / 'README.md').write_text('Finished per-role outputs of the full Assistant-Axis run (see full/orchestrate.py).\n')
    for user in ACCOUNTS:
        meta = {'title': 'aa-full-state', 'id': f'{user}/aa-full-state', 'licenses': [{'name': 'CC0-1.0'}]}
        json.dump(meta, open(STATE / 'dataset-metadata.json', 'w'))
        exists = kq(f'kaggle datasets status {user}/aa-full-state', user)
        if 'ready' in exists or 'pending' in exists:
            sh(f'kaggle datasets version -p {STATE} -m "state update" --dir-mode zip', check=False, user=user)
        else:
            sh(f'kaggle datasets create -p {STATE} --dir-mode zip', check=False, user=user)


def done_roles():
    return {f.stem for f in (STATE / 'activations').glob('*.pt')} if (STATE / 'activations').exists() else set()


def progress():  # any saved stage counts as progress (c01 v2 generated 24 roles but its judge crashed)
    return sum(len(list((STATE / d).glob('*'))) for d in ('responses', 'scores', 'activations') if (STATE / d).exists())


def _has(kind, r):
    ext = {'responses': '.jsonl', 'scores': '.json', 'activations': '.pt'}[kind]
    f = STATE / kind / f'{r}{ext}'
    return f.exists() and (kind != 'scores' or f.stat().st_size > 2)   # {} = judge died, not scored


def needs_score(r):
    return not r.startswith('default')


def runnable(n):
    """A chunk is worth a GPU session if some role still needs generating, or has scores but no activations."""
    return any(not _has('responses', r) or (not _has('activations', r) and (_has('scores', r) or not needs_score(r)))
               for r in chunks()[n])


def awaiting_judge():
    return [r for rs in chunks() for r in rs if _has('responses', r) and needs_score(r) and not _has('scores', r)]


def _locked_upload():
    with locked():
        upload_state(); write_status()


def write_status():
    """full/runs/STATUS.md: per chunk what is generated / judged / done, and which account is running what."""
    c = load_claims(); rows = []
    cnt = lambda kind, rs: sum(_has(kind, r) for r in rs)
    for n, rs in enumerate(chunks()):
        g, j, d = cnt('responses', rs), sum(_has('scores', r) or not needs_score(r) for r in rs), cnt('activations', rs)
        run = c['running'].get(str(n))
        if d == len(rs): state = 'done'
        elif run: state = f"running on {run['account']} since {run['since']}"
        elif c['stalls'].get(str(n), 0) >= 2: state = 'skipped after 2 failed runs'
        elif g > j: state = 'waiting for local judging'
        elif g == len(rs): state = 'needs activations pass'
        elif g: state = 'partly generated; queued'
        else: state = 'queued'
        rows.append(f'| c{n:02d} | {rs[0]} .. {rs[-1]} | {g}/{len(rs)} | {j}/{len(rs)} | {d}/{len(rs)} | {state} |')
    done = len(done_roles())
    acc = [f"- **{u}**: " + (f"running c{int(k):02d}" if (k := next((k for k, v in c['running'].items() if v['account'] == u), None)) is not None
                              else 'idle') + (f" — last: {c['accounts'][u]}" if u in c['accounts'] else '') for u in ACCOUNTS]
    STATUS_MD.write_text('\n'.join([
        f'# Full run status — {time.strftime("%Y-%m-%d %H:%M")} IST', '',
        f'**{done}/276 roles done** (activations saved); {276 - done} left.', '', '## Accounts', *acc, '',
        '## Chunks', '| chunk | roles | generated | judged | done | state |', '|---|---|---|---|---|---|', *rows, '',
        'generated = answers saved; judged = scored by blind Claude judges (or no score needed); '
        'done = activations saved. Rewritten by full/orchestrate.py on every event.', '']))


def summary():
    done = done_roles()
    for n, rs in enumerate(chunks()):
        miss = [r for r in rs if r not in done]
        print(f'c{n:02d}: {len(rs) - len(miss):2d}/{len(rs)} done' + (f'  missing: {" ".join(miss)}' if miss and len(miss) < len(rs) else ''))
    print(f'TOTAL {len(done)}/276 roles complete')


def kstatus(n):
    out = kq(f'kaggle kernels status {kid(n)}')
    for s in ('RUNNING', 'QUEUED', 'COMPLETE', 'ERROR', 'CANCEL'):
        if s in out: return s
    return 'NONE' if '404' in out or 'not found' in out.lower() else 'UNKNOWN'


def wait_state_ready():
    for _ in range(60):
        if kq(f'kaggle datasets status {USER}/aa-full-state').endswith('ready'): return
        time.sleep(20)


def autopilot():
    stamp = lambda: time.strftime('%Y-%m-%d %H:%M')
    say = lambda m: print(f'{stamp()} [{USER}] {m}', flush=True)
    def note(m):
        with locked():
            c = load_claims(); c['accounts'][USER] = f'{stamp()} {m}'; save_claims(c); write_status()
    with locked():
        c = load_claims(); mine = [k for k, v in c['running'].items() if v['account'] == USER]
    cur = int(mine[0]) if mine else None
    while True:
        if cur is not None:
            s = kstatus(cur)
            if s in ('RUNNING', 'QUEUED', 'UNKNOWN'):
                time.sleep(600); continue
            say(f'c{cur:02d} ended: {s}')
            with locked():
                before, pbefore = len(done_roles()), progress()
                collect([cur]); upload_state(); wait_state_ready()
                gained = len(done_roles()) - before
                c = load_claims()
                c['stalls'][str(cur)] = 0 if gained or progress() > pbefore else c['stalls'].get(str(cur), 0) + 1
                c['running'].pop(str(cur), None); save_claims(c)
            say(f'c{cur:02d} added {gained} roles; total {len(done_roles())}/276')
            note(f'c{cur:02d} ended {s}, +{gained} roles'); cur = None
        with locked():
            c = load_claims(); done = done_roles()
            left = [n for n, rs in enumerate(chunks()) if any(r not in done for r in rs) and c['stalls'].get(str(n), 0) < 2]
            free = [n for n in left if runnable(n) and str(n) not in c['running']]
            if free:
                n = free[0]; c['running'][str(n)] = {'account': USER, 'since': stamp()}; save_claims(c)
        if not left:
            say(f'AUTOPILOT FINISHED: {len(done)}/276 roles complete'); note('finished'); return
        if not free:   # everything left is running elsewhere or waiting for local judging
            note('idle: nothing free (rest running on the other account or waiting for judging)'); time.sleep(600); continue
        msg = kq(f'kaggle kernels push -p {kdir(n)}')
        if 'successfully pushed' in msg:
            say(f'pushed c{n:02d}'); note(f'pushed c{n:02d}'); cur = n; time.sleep(120)
        else:
            with locked():
                c = load_claims(); c['running'].pop(str(n), None); save_claims(c)
            say(f'push c{n:02d} refused: {msg[-160:]}'); note(f'push refused: {msg[-90:]}'); time.sleep(900)


if __name__ == '__main__':
    cmd, args = sys.argv[1], [int(a) for a in sys.argv[2:]]
    {'build': build, 'status': status, 'summary': summary, 'upload-state': lambda: _locked_upload(), 'autopilot': autopilot,
     'status-file': lambda: (write_status(), print(STATUS_MD.read_text()))}.get(cmd, lambda: None)() \
        if cmd not in ('push', 'collect') else {'push': push, 'collect': collect}[cmd](args)
