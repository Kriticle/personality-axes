# Activation capping for Qwen3-8B on Kaggle T4 x2: build the capping config, then evaluate it.
# Rendered by capping/build_kernel.py -- edit the template / capping_core.py, not capping/kernel/run.py.
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'   # the split 8B model hits async "unspecified launch failure" otherwise (kernel_debug/)
import subprocess, sys, json, time, glob, shutil, random
T0 = time.time()
log = lambda *a: print(f'[{(time.time() - T0) / 3600:5.2f}h]', *a, flush=True)
EVAL_CONFIGS = __EVAL_CONFIGS__
PRESSURE_ROLES = __PRESSURE_ROLES__
QUESTIONS = __QUESTIONS__          # held-out question indices (the axis used the first 16 / 24)
PRESSURE_PROMPTS = [0, 1]
BENIGN_PROMPTS = [1, 2, 3, 4]      # default.json: 'You are an AI assistant.', 'You are a large language model.', ...
PROBE_LAYERS = [18, 33]
GEN_BATCH = 8

r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'vllm==0.10.2', 'transformers==4.57.1',
                    'jsonlines', 'python-dotenv', 'openai==1.99.9'], capture_output=True, text=True)
print(r.stderr[-800:]); assert r.returncode == 0, 'pip install failed'

# ---- setup copied from kernel/00_calibrate.ipynb (patches + helpers) ----
__SETUP__
# ---- end setup ----
open('/kaggle/working/capping_core.py', 'w').write(__CORE__)
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
