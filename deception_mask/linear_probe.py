"""Trained linear probe (L2 logistic regression) for lie vs honest on the MASK run.

The pre-registered analysis used a difference-of-means direction. This asks whether an *optimised* linear
direction does better, under the same controls:
  * 5 folds grouped by scenario (same assignment as analyze()), scores are out-of-fold;
  * headline metric = within-prompt AUROC (lie vs honest answers to the same prompt);
  * baselines: a logistic model on [length, entropy] alone, and activations + [length, entropy];
  * control task: the same probe trained on shuffled labels must score ~0.5.
A probe IS a direction (its weight vector); this tests readability, not causal use.

    python deception_mask/linear_probe.py --out deception_mask/out_colab
"""
import argparse, json, random, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import mask_deception as m


def fit_predict(Xtr, ytr, Xte, lam, iters=150):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = torch.zeros(Xtr.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, ytr) + lam * (w @ w)
        loss.backward(); return loss
    opt.step(closure)
    return (Xte @ w + b).detach(), w.detach() / sd   # weight mapped back to raw activation space


def oof(X, y, fold, lam):
    s = torch.zeros(len(y))
    for f in range(5):
        te = fold == f
        s[te], _ = fit_predict(X[~te], y[~te], X[te], lam)
    return s


def scores(rows, s):
    lies = [float(v) for r, v in zip(rows, s) if r['label'] == 'lie']
    hon = [float(v) for r, v in zip(rows, s) if r['label'] == 'honest']
    idx = {r['key']: float(v) for r, v in zip(rows, s)}
    return m.auroc(lies, hon), m.within_prompt_auroc(rows, lambda r: idx[r['key']])[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='deception_mask/out_colab')
    ap.add_argument('--layers', default='all'); a = ap.parse_args()
    out = Path(a.out)
    items = json.load(open(out / 'items.json')); lab = json.load(open(out / 'labels.json'))
    for it in items:
        if 'A' in it: it['verdict'] = lab[it['key']]
    m.label(items, strict=False)
    acts = torch.load(out / 'acts.pt')
    LH = [it for it in items if it['kind'] == 'pressure' and it.get('label') in ('lie', 'honest') and it['key'] in acts]
    tasks = sorted({r['task'] for r in LH}); random.Random(0).shuffle(tasks)
    fold = torch.tensor([{t: i % 5 for i, t in enumerate(tasks)}[r['task']] for r in LH])
    y = torch.tensor([r['label'] == 'lie' for r in LH], dtype=torch.float32)
    ysh = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(0))]
    base = torch.tensor([[float(r['n_tokens']), r['entropy']] for r in LH])
    A = torch.stack([acts[r['key']].float() for r in LH])            # (N, L, D)
    L = A.shape[1]
    layers = range(L) if a.layers == 'all' else [int(x) for x in a.layers.split(',')]
    print(f'{len(LH)} answers ({int(y.sum())} lies) from {len(tasks)} scenarios; {L} layers')

    # regularisation chosen once, at the pre-registered layer, by out-of-fold within-prompt AUROC
    lam_grid = [1e-4, 1e-3, 1e-2, 1e-1]
    lam_scores = {lam: scores(LH, oof(A[:, 22], y, fold, lam))[1] for lam in lam_grid}
    lam = max(lam_scores, key=lam_scores.get)
    print('lambda search at layer 22:', {k: round(v, 3) for k, v in lam_scores.items()}, '-> using', lam)

    res = {'n': len(LH), 'n_lie': int(y.sum()), 'lambda': lam, 'lambda_search_L22': lam_scores}
    res['baseline_len_entropy'] = dict(zip(('auroc', 'within_prompt'), scores(LH, oof(base, y, fold, 1e-3))))
    print('baseline (length + entropy only):', {k: round(v, 3) for k, v in res['baseline_len_entropy'].items()})
    res['by_layer'] = {}
    for l in layers:
        X = A[:, l]
        r = {'probe': scores(LH, oof(X, y, fold, lam)),
             'probe_plus_len_entropy': scores(LH, oof(torch.cat([X, base], 1), y, fold, lam)),
             'shuffled_labels_control': scores(LH, oof(X, ysh, fold, lam))}
        res['by_layer'][l] = {k: dict(zip(('auroc', 'within_prompt'), v)) for k, v in r.items()}
        print(f"L{l:2d}  probe within-prompt {r['probe'][1]:.3f} (overall {r['probe'][0]:.3f})   "
              f"+len/ent {r['probe_plus_len_entropy'][1]:.3f}   shuffled control {r['shuffled_labels_control'][1]:.3f}")
    best = max(res['by_layer'], key=lambda l: res['by_layer'][l]['probe']['within_prompt'])
    res['best_layer'] = best
    _, w = fit_predict(A[:, best], y, A[:1, best], lam)
    torch.save({'layer': best, 'w': w, 'lambda': lam}, out / 'linear_probe_direction.pt')
    json.dump(res, open(out / 'linear_probe.json', 'w'), indent=1)
    print('best layer', best, res['by_layer'][best]['probe'])


if __name__ == '__main__':
    main()
