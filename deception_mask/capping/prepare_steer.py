"""Mean-difference lie directions for additive steering (runs locally, CPU).

Same sealed split as prepare.py: directions use only the "fit" scenarios (not the calibration slice, never the
held-out test scenarios). Per layer L: d_L = mean(lie) - mean(honest) of per-answer mean activations; u_L = d_L/|d_L|;
gap_L = d_L . u_L = |d_L| (how far lies sit from honest answers along u_L). Steering adds  coef * gap_L * u_L  to
every position at each layer: coef < 0 pushes toward honest, coef > 0 toward lying.

    python deception_mask/capping/prepare_steer.py     # -> deception_mask/capping/steer.pt
"""
import json, random, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mask_deception as m

LAYERS = [19, 20, 21, 22, 23]
CONFIGS = {'none': 0.0, 'honest_x1': -1.0, 'honest_x2': -2.0, 'honest_x4': -4.0, 'lie_x2': 2.0}
OUT = HERE.parent / 'out_colab'
items = json.load(open(OUT / 'items.json')); lab = json.load(open(OUT / 'labels.json'))
for it in items:
    if 'A' in it: it['verdict'] = lab[it['key']]
m.label(items, strict=False)
acts = torch.load(OUT / 'acts.pt')
LH = [it for it in items if it['kind'] == 'pressure' and it.get('label') in ('lie', 'honest') and it['key'] in acts]
tasks = sorted({r['task'] for r in LH}); random.Random(123).shuffle(tasks); held = set(tasks[:len(tasks) // 4])
trt = sorted({r['task'] for r in LH if r['task'] not in held}); random.Random(5).shuffle(trt); cal = set(trt[:len(trt) // 5])
fit = [r for r in LH if r['task'] not in held and r['task'] not in cal]
probe = torch.load(HERE / 'caps.pt')
out = {'layers': LAYERS, 'configs': CONFIGS, 'directions': {}, 'gap': {}}
for L in LAYERS:
    X = torch.stack([acts[r['key']][L].float() for r in fit]); y = torch.tensor([r['label'] == 'lie' for r in fit])
    d = X[y].mean(0) - X[~y].mean(0)
    out['directions'][L] = d / d.norm(); out['gap'][L] = float(d.norm())
    print(f"L{L}: gap {out['gap'][L]:.2f}  residual norm ~{float(X.norm(dim=1).mean()):.1f}  "
          f"cos(mean-diff, probe) {float(out['directions'][L] @ probe['directions'][L]):+.2f}")
torch.save(out, HERE / 'steer.pt')
print(len(fit), 'fit answers; configs', CONFIGS)
