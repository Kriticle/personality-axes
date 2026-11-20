"""Generate MASK pressure answers from Qwen3-8B with and without capping along the lie-probe directions.

Cap at layer L (applied to every position of the residual stream after that decoder layer):
    h <- h - max(0, h.u_L - tau_L) * u_L        (u_L = unit lie-probe direction, tau_L from honest answers)
so the model can move freely toward "honest" but not past tau_L toward "lie". Same rule as the Assistant-axis
capping, with the lie probe instead of -axis.

Configs: none | loose (tau = honest p90) | tight (tau = honest p50). Every config uses the same sampling seeds.

    python lie_cap.py --caps caps.pt --prompts prompts.json --out OUT [--k 4] [--bs 16] [--model Qwen/Qwen3-8B]
"""
import argparse, json, os, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Capper:
    def __init__(self, model, caps, mode):
        self.handles, self.n_clamped, self.n_tok = [], 0, 0
        if mode == 'none':
            return
        layers = model.model.layers
        for L in caps['layers']:
            u = caps['directions'][L].to(model.device, torch.float32)
            tau = caps['tau'][mode][L]
            def hook(_m, _i, out, u=u, tau=tau):
                h = out[0] if isinstance(out, tuple) else out
                p = h.float() @ u
                ex = (p - tau).clamp(min=0)
                if h.shape[1] == 1:                       # decode step: count clamped generated tokens
                    self.n_clamped += int((ex > 0).sum()); self.n_tok += ex.numel()
                h.sub_((ex[..., None] * u).to(h.dtype))
                return out
            self.handles.append(layers[L].register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()


def render(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def run(model, tok, prompts, k, bs, max_new, seed):
    jobs = [(p, j) for p in prompts for j in range(k)]
    jobs.sort(key=lambda pj: len(render(tok, pj[0]['msgs'])))
    out = []
    for i in range(0, len(jobs), bs):
        batch = jobs[i:i + bs]
        enc = tok([render(tok, p['msgs']) for p, _ in batch], return_tensors='pt', padding=True).to(model.device)
        torch.manual_seed(seed + i)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.9,
                             pad_token_id=tok.pad_token_id)
        for (p, j), g in zip(batch, gen[:, enc.input_ids.shape[1]:]):
            out.append(dict(key=f"{p['task']}/cap/{j}", task=p['task'], subset=p['subset'], A=p['A'], B=p['B'],
                            belief=p['belief'], msgs=p['msgs'], answer=tok.decode(g, skip_special_tokens=True).strip()))
        print(f'  {i + len(batch)}/{len(jobs)}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--caps', required=True); ap.add_argument('--prompts', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='Qwen/Qwen3-8B'); ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--bs', type=int, default=16); ap.add_argument('--max_new', type=int, default=256)
    ap.add_argument('--configs', default='none,loose,tight'); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    caps = torch.load(a.caps); prompts = json.load(open(a.prompts))
    if a.limit: prompts = prompts[:a.limit]
    tok = AutoTokenizer.from_pretrained(a.model, padding_side='left')
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dt = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(dev).eval()
    for cfg in a.configs.split(','):
        f = out / f'answers_{cfg}.json'
        if f.exists():
            print('skip', cfg, '(done)'); continue
        t = time.time(); cap = Capper(model, caps, cfg)
        ans = run(model, tok, prompts, a.k, a.bs, a.max_new, seed=1234)
        cap.remove()
        meta = dict(config=cfg, seconds=time.time() - t,
                    frac_generated_tokens_clamped_per_layer=(cap.n_clamped / cap.n_tok) if cap.n_tok else None)
        json.dump(dict(meta=meta, answers=ans), open(f, 'w'))
        print(cfg, meta, flush=True)


if __name__ == '__main__':
    main()
