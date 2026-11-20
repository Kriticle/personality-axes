"""Preliminary persona map from the finished roles of the full run, projected on the calibration axis.
python figures/compute_persona_map.py  (needs torch; reads full/state/activations, out_v8/axis.pt)
-> results/persona_map_partial.json"""
import glob, json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
LAYER, MIN_SCORE3 = 18, 10          # middle layer of 36 (the layer the full run keeps per-sample activations for)
axis = torch.load(ROOT / 'out_v8/axis.pt', weights_only=False).float()
a = axis[LAYER] / axis[LAYER].norm()
roles, vecs, rate = [], [], {}
for f in sorted(glob.glob(str(ROOT / 'full/state/activations/*.pt'))):
    d = torch.load(f, weights_only=False)
    sc = [v for v in d['scores'].values() if v is not None]
    rate[d['role']] = sum(v == 3 for v in sc) / len(sc) if sc else None
    if d['n_score3'] >= MIN_SCORE3:
        roles.append(d['role']); vecs.append(d['mean_score3'][LAYER].float())
X = torch.stack(vecs)
proj = (X @ a).tolist()
Xc = X - X.mean(0)
U, S, V = torch.linalg.svd(Xc, full_matrices=False)
pc = Xc @ V[:2].T
cos1 = float(V[0] @ a)
if cos1 < 0:                         # orient PC1 so that + = more Assistant-like
    pc[:, 0] *= -1; cos1 = -cos1
out = dict(layer=LAYER, min_score3=MIN_SCORE3, n_roles=len(roles), cos_pc1_calib_axis=cos1,
           var_explained=(S[:5] ** 2 / (S ** 2).sum()).tolist(),
           roles=[dict(role=r, proj=p, pc1=float(x), pc2=float(y)) for r, p, (x, y) in zip(roles, proj, pc.tolist())],
           frac_score3=rate, excluded_few_score3=sorted(set(rate) - set(roles)))
json.dump(out, open(ROOT / 'results/persona_map_partial.json', 'w'), indent=1)
print(len(roles), 'roles; cos(PC1, calibration axis) =', round(cos1, 3), '; var PC1..3', [round(v, 3) for v in out['var_explained'][:3]],
      '; excluded', out['excluded_few_score3'])
