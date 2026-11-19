# Diagnose CUDA "unspecified launch failure" in 2_activations.py on Kaggle T4 x2.
import subprocess, sys, os, json, time, glob, zipfile, shutil
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'vllm==0.10.2', 'transformers==4.57.1', 'jsonlines', 'python-dotenv', 'openai==1.99.9'], capture_output=True, text=True)
print(r.stderr[-800:]); assert r.returncode == 0
subprocess.run('nvidia-smi', shell=True)

# ---- setup cell copied verbatim from 00_calibrate.ipynb ----
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'  # Xet token endpoint rate-limits (429) anonymous Kaggle IPs
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '60')
import glob, os, shutil, subprocess, time, json, zipfile
hits = glob.glob('/kaggle/input/**/pipeline/1_generate.py', recursive=True)
if not hits:
    zips = glob.glob('/kaggle/input/**/assistant-axis-code*.zip', recursive=True)
    assert zips, 'Add the assistant-axis-code dataset to this notebook'
    zipfile.ZipFile(zips[0]).extractall('/kaggle/working/_src'); hits = glob.glob('/kaggle/working/_src/**/pipeline/1_generate.py', recursive=True)
SRC = os.path.dirname(os.path.dirname(hits[0]))
AA = '/kaggle/working/aa'
if not os.path.exists(AA): shutil.copytree(SRC, AA)
_g = f'{AA}/assistant_axis/generation.py'; _s = open(_g).read()
if 'VLLM_DTYPE' not in _s:
    _s = _s.replace("trust_remote_code=True,\n        )", "trust_remote_code=True,\n            dtype=os.environ.get('VLLM_DTYPE', 'auto'),\n        )", 1)
    if 'import os' not in _s: _s = 'import os\n' + _s
    open(_g, 'w').write(_s)
assert 'VLLM_DTYPE' in open(_g).read(), 'dtype patch did not apply'
os.environ['VLLM_DTYPE'] = 'half'
_m = f'{AA}/assistant_axis/internals/model.py'; _s = open(_m).read()
_anchor = "self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)\n"
if 'NO_OFFLOAD' not in _s:
    assert _anchor in _s, 'model.py anchor not found'
    _s = _s.replace(_anchor, _anchor + "        # NO_OFFLOAD: fail fast instead of silently running layers on CPU/disk\n"
        "        _bad = {k: v for k, v in (getattr(self.model, 'hf_device_map', None) or {}).items() if str(v) in ('cpu', 'disk')}\n"
        "        if _bad: raise RuntimeError(f'model offloaded to CPU/disk (GPUs not free?): {list(_bad)[:5]}')\n", 1)
    open(_m, 'w').write(_s)
assert 'NO_OFFLOAD' in open(_m).read(), 'offload guard did not apply'
# transformers 5.x returns BatchEncoding from apply_chat_template(tokenize=True); force plain token lists
for _f in glob.glob(f'{AA}/assistant_axis/**/*.py', recursive=True):
    _s = open(_f).read()
    if 'tokenize=True,' in _s and 'return_dict=False' not in _s:
        open(_f, 'w').write(_s.replace('tokenize=True,', 'tokenize=True, return_dict=False,'))
def gpu_used_mib():
    q = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return [int(x) for x in q.stdout.split()]
def wait_gpus_free(limit_mib=1500, timeout=300):
    for _ in range(timeout // 5):
        u = gpu_used_mib()
        if max(u) < limit_mib: print('GPUs free:', u, 'MiB used'); return
        time.sleep(5)
    subprocess.run('nvidia-smi', shell=True)
    raise RuntimeError(f'GPUs still busy: {gpu_used_mib()} MiB used')
PIPE = f'{AA}/pipeline'
OUT = '/kaggle/working/calib'; os.makedirs(OUT, exist_ok=True)
REPORT = {}
TARGET = 'Qwen/Qwen3-8B'; JUDGE = 'Qwen/Qwen3-14B-AWQ'; TARGET_LAYER = 18
ROLES = ['default', 'assistant', 'pirate', 'demon', 'oracle', 'accountant']
QCOUNT = 16
def run(cmd, env=None):
    t = time.time(); print('$', cmd)
    r = subprocess.run(cmd, shell=True, cwd=PIPE, env={**os.environ, **(env or {})})
    assert r.returncode == 0, f'failed: {cmd}'
    return time.time() - t
print(SRC)
# ---- end setup ----
# model.py: allow choosing attention implementation via env
_m = f'{AA}/assistant_axis/internals/model.py'; _s = open(_m).read()
if 'AA_ATTN' not in _s:
    _s = _s.replace('"dtype": dtype,\n        }', '"dtype": dtype,\n        }\n        if os.environ.get("AA_ATTN"): model_kwargs["attn_implementation"] = os.environ["AA_ATTN"]\n        if os.environ.get("AA_MAXMEM"): model_kwargs["max_memory"] = {0: os.environ["AA_MAXMEM"], 1: os.environ["AA_MAXMEM"]}', 1)
    _s = _s.replace('\nimport torch\n', '\nimport os\nimport torch\n', 1)
    open(_m, 'w').write(_s)
assert 'AA_ATTN' in open(_m).read()

RESULTS = {}
# ---- 1. P2P check ----
P2P = r"""
import torch
print('can_access_peer 0->1', torch.cuda.can_device_access_peer(0, 1), '1->0', torch.cuda.can_device_access_peer(1, 0))
bad = 0
for i in range(10):
    x = torch.randn(64, 1024, 1024, device='cuda:0', dtype=torch.float16)
    y = x.to('cuda:1'); z = y.to('cuda:0'); torch.cuda.synchronize()
    ok = torch.equal(x.cpu(), y.cpu()) and torch.equal(x, z)
    bad += not ok
print('p2p copy mismatches:', bad, '/ 10')
"""
p = subprocess.run([sys.executable, '-c', P2P], capture_output=True, text=True)
print(p.stdout, p.stderr[-2000:]); RESULTS['p2p'] = (p.returncode, p.stdout.strip(), p.stderr[-500:])

# ---- 2. activation variants on the role that crashed ----
RESP = glob.glob('/kaggle/input/**/aa-calib-responses/**/responses', recursive=True) or glob.glob('/kaggle/input/**/responses', recursive=True)
SCORES = glob.glob('/kaggle/input/**/aa-calib-responses/**/scores', recursive=True) or glob.glob('/kaggle/input/**/scores', recursive=True)
print('inputs:', RESP, SCORES); assert RESP and SCORES
VARIANTS = [
    ('baseline_blocking', 8, {'CUDA_LAUNCH_BLOCKING': '1'}),
    ('eager_attn',        8, {'AA_ATTN': 'eager'}),
    ('batch2',            2, {}),
]
for name, bs, env in VARIANTS:
    out = f'/kaggle/working/dbg/{name}'; shutil.rmtree(out, ignore_errors=True)
    wait_gpus_free()
    cmd = (f"python 2_activations.py --model {TARGET} --responses_dir {RESP[0]} --output_dir {out} --roles accountant pirate "
           f"--dtype float16 --save_dtype float16 --scores_dir {SCORES[0]} --keep_layers {TARGET_LAYER} --batch_size {bs}")
    t = time.time(); print('\n=====', name, cmd, env, flush=True)
    try:
        r = subprocess.run(cmd, shell=True, cwd=PIPE, env={**os.environ, **env}, capture_output=True, text=True, timeout=1500)
        rc, err = r.returncode, r.stderr
    except subprocess.TimeoutExpired as e:
        rc, err = 'timeout', str(e.stderr)[-3000:]
    saved = sorted(os.path.basename(f) for f in glob.glob(f'{out}/*.pt'))
    tail = '\n'.join(l for l in err.splitlines() if 'DEBUG' not in l and 'Loading checkpoint' not in l)[-3000:]
    print(name, 'rc', rc, 'saved', saved, 'secs', round(time.time() - t)); print(tail, flush=True)
    RESULTS[name] = {'rc': rc, 'saved': saved, 'secs': time.time() - t, 'tail': tail}
    if saved and 'pirate.pt' in saved:
        import torch
        d = torch.load(f'{out}/pirate.pt', weights_only=False)
        RESULTS[name]['finite'] = bool(torch.isfinite(d['mean_all']).all()); print('finite:', RESULTS[name]['finite'])
json.dump(RESULTS, open('/kaggle/working/debug_results.json', 'w'), indent=1, default=str)
print(json.dumps({k: (v if k == 'p2p' else {kk: vv for kk, vv in v.items() if kk != 'tail'}) for k, v in RESULTS.items()}, indent=1, default=str))
