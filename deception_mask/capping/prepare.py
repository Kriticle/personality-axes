"""Prepare the lie-capping experiment (runs locally, CPU).

Directions: L2 logistic-regression lie probes at layers 19-23, trained only on the 75% training scenarios of the
sealed split used by linear_probe.py --confirm (seed 123), lambda 0.01 (what that protocol chose).
The probe fits its own training data perfectly, so caps are NOT set on it: the training scenarios are split again
(seed 5) into fit (80%) and calibration (20%) scenarios; the direction is fit on "fit" only.
Caps: tau_L = quantile of the CALIBRATION scenarios' honest answers' projections on the unit probe direction, so a capped
model can do anything a typical honest answer does but cannot go further toward "lie":
    loose = 90th percentile of honest, tight = 50th percentile of honest.
Prompts: the pressure prompts of the sealed held-out scenarios that have a stable belief (never seen by the probe).

    python deception_mask/capping/prepare.py      # -> deception_mask/capping/caps.pt, prompts.json
"""
import json, random, sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mask_deception as m
import linear_probe as lp

LAYERS, LAM, SPLIT_SEED = [19, 20, 21, 22, 23], 0.01, 123
OUT = HERE.parent / 'out_colab'

items = json.load(open(OUT / 'items.json')); lab = json.load(open(OUT / 'labels.json'))
for it in items:
    if 'A' in it: it['verdict'] = lab[it['key']]
bel = m.label(items, strict=False)
acts = torch.load(OUT / 'acts.pt')
LH = [it for it in items if it['kind'] == 'pressure' and it.get('label') in ('lie', 'honest') and it['key'] in acts]
tasks = sorted({r['task'] for r in LH}); random.Random(SPLIT_SEED).shuffle(tasks)
held = set(tasks[:len(tasks) // 4])
trt = sorted({r['task'] for r in LH if r['task'] not in held}); random.Random(5).shuffle(trt)
cal_t = set(trt[:len(trt) // 5])
tr = [r for r in LH if r['task'] not in held and r['task'] not in cal_t]
cal = [r for r in LH if r['task'] in cal_t]
y = torch.tensor([r['label'] == 'lie' for r in tr], dtype=torch.float32)
yc = torch.tensor([r['label'] == 'lie' for r in cal], dtype=torch.float32)
print(len(tr), 'fit answers,', len(cal), 'calibration answers')

caps = {'layers': LAYERS, 'lambda': LAM, 'directions': {}, 'tau': {'loose': {}, 'tight': {}}, 'calib_auroc': {}}
for L in LAYERS:
    X = torch.stack([acts[r['key']][L].float() for r in tr])
    _, w = lp.fit_predict(X, y, X[:1], LAM)
    u = w / w.norm()
    Xc = torch.stack([acts[r['key']][L].float() for r in cal])
    p = Xc @ u
    hon = p[yc == 0]
    caps['directions'][L] = u
    caps['tau']['loose'][L] = float(torch.quantile(hon, 0.90))
    caps['tau']['tight'][L] = float(torch.quantile(hon, 0.50))
    caps['calib_auroc'][L] = m.auroc(p[yc == 1].tolist(), hon.tolist())
    print(f"L{L}: calibration AUROC {caps['calib_auroc'][L]:.3f}  honest p50 {caps['tau']['tight'][L]:.2f}  "
          f"p90 {caps['tau']['loose'][L]:.2f}  lie p50 {float(p[yc == 1].median()):.2f}")
torch.save(caps, HERE / 'caps.pt')

# held-out pressure prompts with a stable belief (one entry per prompt; K samples drawn on Colab)
seen, prompts = set(), []
for it in items:
    if it['kind'] == 'pressure' and it['task'] in held and bel.get(it['task']) and it['task'] not in seen:
        seen.add(it['task'])
        prompts.append(dict(task=it['task'], subset=it['subset'], A=it['A'], B=it['B'], belief=bel[it['task']],
                            msgs=it['msgs']))
json.dump(prompts, open(HERE / 'prompts.json', 'w'))
print(len(prompts), 'held-out prompts with a belief;', dict(__import__('collections').Counter(p['subset'] for p in prompts)))

# benign check: 40 ordinary questions (no system prompt), not used anywhere else in the MASK run
qs = [json.loads(l)['question'] for l in open(HERE.parent.parent / 'mini' / 'extraction_questions.jsonl')]
used = {it['msgs'][-1]['content'] for it in items if it['kind'] == 'control'}
benign = [q for q in qs if q not in used]
benign = random.Random(9).sample(benign, 40)
json.dump([dict(task=f'benign/{i}', subset='benign', A='', B='', belief='', msgs=[{'role': 'user', 'content': q}])
           for i, q in enumerate(benign)], open(HERE / 'benign.json', 'w'))
print('40 benign questions ->', HERE / 'benign.json')
