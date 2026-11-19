# Activation capping for Qwen3-8B on Kaggle T4 x2: build the capping config, then evaluate it.
# Rendered by capping/build_kernel.py -- edit the template / capping_core.py, not capping/kernel/run.py.
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'   # the split 8B model hits async "unspecified launch failure" otherwise (kernel_debug/)
import subprocess, sys, json, time, glob, shutil, random
T0 = time.time()
log = lambda *a: print(f'[{(time.time() - T0) / 3600:5.2f}h]', *a, flush=True)
EVAL_CONFIGS = [None, 'layers_26:30-p0.25', 'layers_22:30-p0.25', 'layers_26:30-p0.5']
PRESSURE_ROLES = ['eldritch', 'leviathan', 'wraith', 'vampire', 'void', 'trickster', 'bard', 'prophet']
QUESTIONS = [200, 201, 202, 203, 204]          # held-out question indices (the axis used the first 16 / 24)
PRESSURE_PROMPTS = [0, 1]
BENIGN_PROMPTS = [1, 2, 3, 4]      # default.json: 'You are an AI assistant.', 'You are a large language model.', ...
PROBE_LAYERS = [18, 33]
GEN_BATCH = 8

r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'vllm==0.10.2', 'transformers==4.57.1',
                    'jsonlines', 'python-dotenv', 'openai==1.99.9'], capture_output=True, text=True)
print(r.stderr[-800:]); assert r.returncode == 0, 'pip install failed'

# ---- setup copied from kernel/00_calibrate.ipynb (patches + helpers) ----
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
open('/kaggle/working/capping_core.py', 'w').write('"""Activation capping for Qwen3-8B (or any model with an Assistant axis from our pipeline).\n\nMirrors the released Qwen3-32B capping config (lu-christina/assistant-axis-vectors, qwen-3-32b/capping_config.pt):\n  * one vector per layer named \'layer_{L}/contrast_role_pos3_default1\' = role(score 3) mean - default mean = -axis[L]\n    (verified: cos(v, axis) = -1.0 and |v| = |axis| at every layer of the released file);\n  * ActivationSteering\'s cap clamps the projection on v_hat from ABOVE at tau, i.e. it puts a floor on how far\n    the residual stream can move away from the Assistant end of the axis;\n  * experiments \'layers_{a}:{b}-p{q}\' cap layers a..b-1 at tau_L = q-quantile of reference projections on v_hat_L.\n    Inferred, not documented upstream: the reference population is per-response mean activations over role +\n    default responses (in the released file the p0.25 cap matches the default vector\'s projection to <0.2 at\n    layers 46/50/53, which fits a mixed population but not a default-only one). We also record per-token quantiles.\n\nEverything here is plain torch/transformers so it runs on Kaggle and on CPU for tests.\n"""\nfrom __future__ import annotations\n\nimport json\nimport math\nfrom pathlib import Path\nfrom typing import Dict, List, Optional\n\nimport torch\n\nVEC_NAME = \'layer_{L}/contrast_role_pos3_default1\'\nPERCENTILES = (0.01, 0.25, 0.5, 0.75)\n\n\ndef layer_modules(model):\n    for get in (lambda m: m.model.layers, lambda m: m.model.language_model.layers, lambda m: m.transformer.h):\n        try:\n            return get(model)\n        except AttributeError:\n            pass\n    raise AttributeError(\'no transformer layers found\')\n\n\ndef _hs(output):\n    return output[0] if isinstance(output, tuple) else output\n\n\ndef chat_ids(tok, conv, add_generation_prompt):\n    return tok.apply_chat_template(conv, tokenize=True, return_dict=False, add_generation_prompt=add_generation_prompt,\n                                   enable_thinking=False)\n\n\ndef response_span(tok, conversation):\n    """Token ids of the full conversation and [start, end) of the final assistant message."""\n    full = chat_ids(tok, conversation, False)\n    prompt = chat_ids(tok, conversation[:-1], True)\n    start = 0\n    while start < min(len(full), len(prompt)) and full[start] == prompt[start]:\n        start += 1\n    end = len(full)\n    # drop the end-of-turn marker(s) after the answer\n    tail = tok.encode(\'<|im_end|>\', add_special_tokens=False) if \'<|im_end|>\' in tok.get_vocab() else []\n    while end > start and tail and full[end - 1] in set(tail + tok.encode(\'\\n\', add_special_tokens=False)):\n        end -= 1\n    return full, start, end\n\n\n@torch.inference_mode()\ndef reference_projections(model, tok, entries: List[dict], axis: torch.Tensor, batch_size: int = 4,\n                          max_len: int = 2048, log=print):\n    """Per-response mean (and per-token) projections onto v_hat_L = -axis_L/|axis_L| for every layer."""\n    layers = layer_modules(model)\n    n_layers = len(layers)\n    vhat = [(-axis[L].float() / axis[L].float().norm()) for L in range(n_layers)]\n    store: Dict[int, torch.Tensor] = {}\n    hooks = []\n    for L, mod in enumerate(layers):\n        def hook(_m, _i, out, L=L):\n            h = _hs(out)\n            store[L] = (h.float() @ vhat[L].to(h.device)).cpu()  # (B, T)\n        hooks.append(mod.register_forward_hook(hook))\n    per_resp = []      # (n, n_layers)\n    per_tok = [[] for _ in range(n_layers)]\n    meta = []\n    try:\n        for b in range(0, len(entries), batch_size):\n            batch = entries[b:b + batch_size]\n            spans = [response_span(tok, e[\'conversation\']) for e in batch]\n            T = min(max_len, max(len(s[0]) for s in spans))\n            ids = torch.full((len(batch), T), tok.pad_token_id or 0, dtype=torch.long)\n            att = torch.zeros((len(batch), T), dtype=torch.long)\n            for i, (full, _, _) in enumerate(spans):\n                full = full[:T]; ids[i, :len(full)] = torch.tensor(full); att[i, :len(full)] = 1\n            dev = next(model.parameters()).device\n            model(input_ids=ids.to(dev), attention_mask=att.to(dev))\n            for i, (full, s, e) in enumerate(spans):\n                e = min(e, T)\n                if e <= s:\n                    continue\n                row = torch.stack([store[L][i, s:e] for L in range(n_layers)])  # (n_layers, t)\n                per_resp.append(row.mean(1))\n                for L in range(n_layers):\n                    per_tok[L].append(row[L].half())\n                meta.append({\'role\': batch[i].get(\'_role\'), \'n_tokens\': e - s})\n            if (b // batch_size) % 25 == 0:\n                log(f\'  reference projections {b + len(batch)}/{len(entries)}\')\n    finally:\n        for h in hooks:\n            h.remove()\n    return torch.stack(per_resp), [torch.cat(t).float() for t in per_tok], meta\n\n\ndef windows(n_layers: int):\n    """Layer windows mirroring the released sweep: 4-wide windows (step 2) over 50-90% depth, plus 8-wide ones."""\n    lo, hi = round(0.5 * n_layers), round(0.9 * n_layers)\n    w4 = [(a, a + 4) for a in range(lo, hi - 3, 2)]\n    w8 = [(a, a + 8) for a in range(lo, hi - 7, 4)]\n    return w4 + w8\n\n\ndef build_config(axis: torch.Tensor, per_resp: torch.Tensor, percentiles=PERCENTILES) -> dict:\n    """Capping config in the released format: {\'vectors\': {...}, \'experiments\': [{\'id\', \'interventions\'}]}."""\n    n_layers = axis.shape[0]\n    vectors = {VEC_NAME.format(L=L): {\'vector\': (-axis[L]).float().clone(), \'layer\': L} for L in range(n_layers)}\n    q = {p: torch.quantile(per_resp.float(), p, dim=0) for p in percentiles}  # each (n_layers,)\n    experiments = []\n    for a, b in windows(n_layers):\n        for p in percentiles:\n            experiments.append({\'id\': f\'layers_{a}:{b}-p{p}\',\n                                \'interventions\': [{\'vector\': VEC_NAME.format(L=L), \'cap\': float(q[p][L])}\n                                                  for L in range(a, b)]})\n    return {\'vectors\': vectors, \'experiments\': experiments}\n\n\ndef experiment(config: dict, exp_id: str) -> dict:\n    for e in config[\'experiments\']:\n        if e[\'id\'] == exp_id:\n            return e\n    raise KeyError(exp_id)\n\n\nclass CapMeter:\n    """Forward hooks that record, on decode steps only, the projection of the generated token\'s residual stream:\n    pre-cap on v_hat at the capped layers (=> fraction clamped) and on axis_hat at probe layers."""\n\n    def __init__(self, model, axis, cap_layers: Dict[int, float], probe_layers: List[int]):\n        self.layers = layer_modules(model)\n        self.cap = cap_layers\n        self.ahat = {L: axis[L].float() / axis[L].float().norm() for L in set(probe_layers) | set(cap_layers)}\n        self.probe = probe_layers\n        self.steps: List[Dict[str, torch.Tensor]] = []\n        self.handles = []\n        self._cur = None\n\n    def __enter__(self):\n        first = min(set(self.cap) | set(self.probe))\n        for L in sorted(set(self.cap) | set(self.probe)):\n            def hook(_m, _i, out, L=L):\n                h = _hs(out)\n                if h.shape[1] != 1:            # prefill: not a generated token\n                    return\n                if L == first:\n                    self._cur = {}; self.steps.append(self._cur)\n                p = (h[:, -1].float() @ self.ahat[L].to(h.device)).cpu()   # projection on axis_hat\n                if L in self.cap:\n                    self._cur[f\'clamp{L}\'] = (-p > self.cap[L])           # v_hat = -axis_hat; clamped iff proj_v > tau\n                if L in self.probe:\n                    self._cur[f\'proj{L}\'] = p\n            self.handles.append(self.layers[L].register_forward_hook(hook))\n        return self\n\n    def __exit__(self, *a):\n        for h in self.handles:\n            h.remove()\n\n    def summarize(self, gen_lengths: List[int]):\n        """Per-sequence stats over its own generated tokens."""\n        out = []\n        for i, n in enumerate(gen_lengths):\n            steps = self.steps[:max(n - 1, 0)]       # decode step k processes generated token k-1\n            rec = {\'n_tokens\': n}\n            if steps:\n                for L in self.probe:\n                    rec[f\'proj{L}\'] = float(torch.stack([s[f\'proj{L}\'][i] for s in steps]).mean())\n                if self.cap:\n                    c = torch.stack([torch.stack([s[f\'clamp{L}\'][i] for L in self.cap]) for s in steps]).float()\n                    rec[\'frac_tokens_clamped_any\'] = float(c.max(1).values.mean())\n                    rec[\'frac_clamped_per_layer\'] = float(c.mean())\n            out.append(rec)\n        return out\n\n\n@torch.inference_mode()\ndef generate(model, tok, convs: List[list], axis, cap_cfg: Optional[dict], exp_id: Optional[str],\n             probe_layers: List[int], max_new_tokens=512, temperature=0.7, top_p=0.9, seed=0):\n    from assistant_axis.steering import build_capping_steerer\n    import contextlib\n    tok.padding_side = \'left\'\n    prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False) for c in convs]\n    enc = tok(prompts, return_tensors=\'pt\', padding=True, add_special_tokens=False)\n    dev = next(model.parameters()).device\n    caps = {}\n    if exp_id:\n        for it in experiment(cap_cfg, exp_id)[\'interventions\']:\n            caps[cap_cfg[\'vectors\'][it[\'vector\']][\'layer\']] = it[\'cap\']\n    torch.manual_seed(seed)\n    steer = build_capping_steerer(model, cap_cfg, exp_id) if exp_id else contextlib.nullcontext()\n    with CapMeter(model, axis, caps, probe_layers) as meter, steer:\n        out = model.generate(**{k: v.to(dev) for k, v in enc.items()}, max_new_tokens=max_new_tokens,\n                             do_sample=temperature > 0, temperature=temperature, top_p=top_p,\n                             pad_token_id=tok.pad_token_id)\n    gen = out[:, enc[\'input_ids\'].shape[1]:].cpu()\n    stops = {tok.eos_token_id, tok.pad_token_id} | set(tok.encode(\'<|im_end|>\', add_special_tokens=False))\n    lengths, texts = [], []\n    for row in gen.tolist():\n        n = next((i + 1 for i, t in enumerate(row) if t in stops), len(row))\n        lengths.append(n)\n        texts.append(tok.decode(row[:n], skip_special_tokens=True).strip())\n    return texts, meter.summarize(lengths)\n')
sys.path.insert(0, '/kaggle/working'); sys.path.insert(0, AA)
import torch
from capping_core import reference_projections, build_config, generate, experiment, PERCENTILES

W = '/kaggle/working/capping'; os.makedirs(W, exist_ok=True)
axis_f = glob.glob('/kaggle/input/**/aa-capping-inputs/**/axis.pt', recursive=True)
assert axis_f, 'add the aa-capping-inputs dataset'
axis = torch.load(axis_f[0], weights_only=False).float(); log('axis', axis_f[0], tuple(axis.shape))
ref_dirs = [d for d in glob.glob('/kaggle/input/**/responses', recursive=True) if glob.glob(f'{d}/*.jsonl')]
log('reference response dirs:', ref_dirs)
entries = []
for d in ref_dirs:
    for f in sorted(glob.glob(f'{d}/*.jsonl')):
        for line in open(f):
            e = json.loads(line); e['_role'] = os.path.basename(f)[:-6]; entries.append(e)
log('reference responses:', len(entries))

from assistant_axis.internals import ProbingModel
pm = ProbingModel(TARGET, dtype=torch.float16)
model, tok = pm.model, pm.tokenizer

# ---- 1. reference projections -> capping config ----
per_resp, per_tok, meta = reference_projections(model, tok, entries, axis, batch_size=4, log=log)
cfg = build_config(axis, per_resp)
torch.save(cfg, f'{W}/capping_config.pt')
roles_arr = [m['role'] for m in meta]
ref = {'n_responses': len(meta), 'n_layers': axis.shape[0],
       'per_response_quantiles': {str(p): torch.quantile(per_resp, p, dim=0).tolist() for p in PERCENTILES},
       'per_token_quantiles': {str(p): [float(torch.quantile(t[torch.randperm(len(t))[:200000]], p)) for t in per_tok] for p in PERCENTILES},
       'role_mean_proj': {r: per_resp[[i for i, x in enumerate(roles_arr) if x == r]].mean(0).tolist() for r in sorted(set(roles_arr))}}
json.dump(ref, open(f'{W}/reference_stats.json', 'w'))
log('capping config:', len(cfg['experiments']), 'experiments; default mean vs p0.25 at L26..29:',
    [(round(ref['role_mean_proj'].get('default', [0] * 99)[L], 1), round(ref['per_response_quantiles']['0.25'][L], 1)) for L in range(26, 30)])

# ---- 2. generation: held-out persona prompts + benign assistant prompts ----
qs = [json.loads(l) for l in open(f'{AA}/data/extraction_questions.jsonl')]
def role_prompts(role):
    d = json.load(open(f'{AA}/data/roles/instructions/{role}.json'))
    return [i['pos'] if isinstance(i, dict) else i for i in d['instruction']]
tasks = []   # (role, prompt_index, question_index, system)
for role in [r for r in PRESSURE_ROLES if os.path.exists(f'{AA}/data/roles/instructions/{r}.json')]:
    ps = role_prompts(role)
    tasks += [(role, p, q, ps[p]) for p in PRESSURE_PROMPTS for q in QUESTIONS]
dps = role_prompts('default')
tasks += [('default', p, q, dps[p].replace('{model_name}', 'Qwen')) for p in BENIGN_PROMPTS for q in QUESTIONS]
log('generation tasks per config:', len(tasks))
stats = {}
for exp_id in EVAL_CONFIGS:
    name = exp_id or 'uncapped'; od = f'{W}/responses/{name}'; os.makedirs(od, exist_ok=True)
    rows = {}
    t = time.time()
    for b in range(0, len(tasks), GEN_BATCH):
        batch = tasks[b:b + GEN_BATCH]
        convs = [([{'role': 'system', 'content': s}] if s else []) + [{'role': 'user', 'content': qs[q]['question']}]
                 for _, _, q, s in batch]
        texts, st = generate(model, tok, convs, axis, cfg, exp_id, PROBE_LAYERS, max_new_tokens=512, seed=b)
        for (role, p, q, s), c, txt, s2 in zip(batch, convs, texts, st):
            rows.setdefault(role, []).append({'system_prompt': s, 'prompt_index': p, 'question_index': q,
                                              'question': qs[q]['question'], 'label': 'pos',
                                              'conversation': c + [{'role': 'assistant', 'content': txt}], 'cap_stats': s2})
    for role, rs in rows.items():
        with open(f'{od}/{role}.jsonl', 'w') as f:
            for x in rs: f.write(json.dumps(x) + '\n')
    stats[name] = {'seconds': time.time() - t}
    log(f'{name}: generated {len(tasks)} in {time.time() - t:.0f}s')

del model, pm; import gc; gc.collect(); torch.cuda.empty_cache()

# ---- 3. judge persona adherence with the pipeline's judge ----
wait_gpus_free()
srv = subprocess.Popen(f"python -m vllm.entrypoints.openai.api_server --model {JUDGE} --tensor-parallel-size 2 "
                       f"--max-model-len 4096 --dtype half --gpu-memory-utilization 0.90 --port 8000",
                       shell=True, stdout=open('/kaggle/working/judge_server.log', 'w'), stderr=subprocess.STDOUT,
                       start_new_session=True)
import urllib.request, signal
for _ in range(180):
    try: urllib.request.urlopen('http://localhost:8000/health'); break
    except Exception: time.sleep(10)
else: raise RuntimeError(open('/kaggle/working/judge_server.log').read()[-3000:])
JENV = {'OPENAI_BASE_URL': 'http://localhost:8000/v1', 'OPENAI_API_KEY': 'EMPTY', 'JUDGE_TEMPERATURE': '0'}
try:
    for exp_id in EVAL_CONFIGS:
        name = exp_id or 'uncapped'
        for _ in range(2):
            run(f"python 3_judge.py --responses_dir {W}/responses/{name} --output_dir {W}/scores/{name} "
                f"--judge_model {JUDGE} --batch_size 64 --requests_per_second 200", JENV)
finally:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try: os.killpg(srv.pid, sig)
        except ProcessLookupError: break
        try: srv.wait(timeout=60); break
        except subprocess.TimeoutExpired: pass
    subprocess.run("pkill -9 -f 'vllm.entrypoints' ; pkill -9 -f 'VLLM::' ", shell=True)

# ---- 4. summary ----
import statistics as stt
summary = {}
for exp_id in EVAL_CONFIGS:
    name = exp_id or 'uncapped'; s = {}
    for kind in ('pressure', 'benign'):
        recs, scores = [], []
        for f in glob.glob(f'{W}/responses/{name}/*.jsonl'):
            role = os.path.basename(f)[:-6]
            if (role == 'default') != (kind == 'benign'): continue
            sc = json.load(open(f'{W}/scores/{name}/{role}.json')) if os.path.exists(f'{W}/scores/{name}/{role}.json') else {}
            for line in open(f):
                e = json.loads(line); recs.append(e['cap_stats'])
                k = f"pos_p{e['prompt_index']}_q{e['question_index']}"
                if k in sc: scores.append(sc[k])
        mean = lambda key: stt.mean([r[key] for r in recs if key in r]) if any(key in r for r in recs) else None
        s[kind] = {'n': len(recs), 'n_scored': len(scores),
                   'judge_mean': stt.mean(scores) if scores else None,
                   'frac_score3': sum(x == 3 for x in scores) / len(scores) if scores else None,
                   **{f'proj{L}': mean(f'proj{L}') for L in PROBE_LAYERS},
                   'frac_tokens_clamped_any': mean('frac_tokens_clamped_any'),
                   'mean_tokens': mean('n_tokens')}
    summary[name] = {**s, **stats.get(name, {})}
json.dump(summary, open(f'{W}/summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
shutil.rmtree('/kaggle/working/aa', ignore_errors=True); shutil.rmtree('/kaggle/working/_src', ignore_errors=True)
log('DONE')
