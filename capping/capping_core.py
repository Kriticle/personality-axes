"""Activation capping for Qwen3-8B (or any model with an Assistant axis from our pipeline).

Mirrors the released Qwen3-32B capping config (lu-christina/assistant-axis-vectors, qwen-3-32b/capping_config.pt):
  * one vector per layer named 'layer_{L}/contrast_role_pos3_default1' = role(score 3) mean - default mean = -axis[L]
    (verified: cos(v, axis) = -1.0 and |v| = |axis| at every layer of the released file);
  * ActivationSteering's cap clamps the projection on v_hat from ABOVE at tau, i.e. it puts a floor on how far
    the residual stream can move away from the Assistant end of the axis;
  * experiments 'layers_{a}:{b}-p{q}' cap layers a..b-1 at tau_L = q-quantile of reference projections on v_hat_L.
    Inferred, not documented upstream: the reference population is per-response mean activations over role +
    default responses (in the released file the p0.25 cap matches the default vector's projection to <0.2 at
    layers 46/50/53, which fits a mixed population but not a default-only one). We also record per-token quantiles.

Everything here is plain torch/transformers so it runs on Kaggle and on CPU for tests.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch

VEC_NAME = 'layer_{L}/contrast_role_pos3_default1'
PERCENTILES = (0.01, 0.25, 0.5, 0.75)


def layer_modules(model):
    for get in (lambda m: m.model.layers, lambda m: m.model.language_model.layers, lambda m: m.transformer.h):
        try:
            return get(model)
        except AttributeError:
            pass
    raise AttributeError('no transformer layers found')


def _hs(output):
    return output[0] if isinstance(output, tuple) else output


def chat_ids(tok, conv, add_generation_prompt):
    return tok.apply_chat_template(conv, tokenize=True, return_dict=False, add_generation_prompt=add_generation_prompt,
                                   enable_thinking=False)


def response_span(tok, conversation):
    """Token ids of the full conversation and [start, end) of the final assistant message."""
    full = chat_ids(tok, conversation, False)
    prompt = chat_ids(tok, conversation[:-1], True)
    start = 0
    while start < min(len(full), len(prompt)) and full[start] == prompt[start]:
        start += 1
    end = len(full)
    # drop the end-of-turn marker(s) after the answer
    tail = tok.encode('<|im_end|>', add_special_tokens=False) if '<|im_end|>' in tok.get_vocab() else []
    while end > start and tail and full[end - 1] in set(tail + tok.encode('\n', add_special_tokens=False)):
        end -= 1
    return full, start, end


@torch.inference_mode()
def reference_projections(model, tok, entries: List[dict], axis: torch.Tensor, batch_size: int = 4,
                          max_len: int = 2048, log=print):
    """Per-response mean (and per-token) projections onto v_hat_L = -axis_L/|axis_L| for every layer."""
    layers = layer_modules(model)
    n_layers = len(layers)
    vhat = [(-axis[L].float() / axis[L].float().norm()) for L in range(n_layers)]
    store: Dict[int, torch.Tensor] = {}
    hooks = []
    for L, mod in enumerate(layers):
        def hook(_m, _i, out, L=L):
            h = _hs(out)
            store[L] = (h.float() @ vhat[L].to(h.device)).cpu()  # (B, T)
        hooks.append(mod.register_forward_hook(hook))
    per_resp = []      # (n, n_layers)
    per_tok = [[] for _ in range(n_layers)]
    meta = []
    try:
        for b in range(0, len(entries), batch_size):
            batch = entries[b:b + batch_size]
            spans = [response_span(tok, e['conversation']) for e in batch]
            T = min(max_len, max(len(s[0]) for s in spans))
            ids = torch.full((len(batch), T), tok.pad_token_id or 0, dtype=torch.long)
            att = torch.zeros((len(batch), T), dtype=torch.long)
            for i, (full, _, _) in enumerate(spans):
                full = full[:T]; ids[i, :len(full)] = torch.tensor(full); att[i, :len(full)] = 1
            dev = next(model.parameters()).device
            model(input_ids=ids.to(dev), attention_mask=att.to(dev))
            for i, (full, s, e) in enumerate(spans):
                e = min(e, T)
                if e <= s:
                    continue
                row = torch.stack([store[L][i, s:e] for L in range(n_layers)])  # (n_layers, t)
                per_resp.append(row.mean(1))
                for L in range(n_layers):
                    per_tok[L].append(row[L].half())
                meta.append({'role': batch[i].get('_role'), 'n_tokens': e - s})
            if (b // batch_size) % 25 == 0:
                log(f'  reference projections {b + len(batch)}/{len(entries)}')
    finally:
        for h in hooks:
            h.remove()
    return torch.stack(per_resp), [torch.cat(t).float() for t in per_tok], meta


def windows(n_layers: int):
    """Layer windows mirroring the released sweep: 4-wide windows (step 2) over 50-90% depth, plus 8-wide ones."""
    lo, hi = round(0.5 * n_layers), round(0.9 * n_layers)
    w4 = [(a, a + 4) for a in range(lo, hi - 3, 2)]
    w8 = [(a, a + 8) for a in range(lo, hi - 7, 4)]
    return w4 + w8


def build_config(axis: torch.Tensor, per_resp: torch.Tensor, percentiles=PERCENTILES) -> dict:
    """Capping config in the released format: {'vectors': {...}, 'experiments': [{'id', 'interventions'}]}."""
    n_layers = axis.shape[0]
    vectors = {VEC_NAME.format(L=L): {'vector': (-axis[L]).float().clone(), 'layer': L} for L in range(n_layers)}
    q = {p: torch.quantile(per_resp.float(), p, dim=0) for p in percentiles}  # each (n_layers,)
    experiments = []
    for a, b in windows(n_layers):
        for p in percentiles:
            experiments.append({'id': f'layers_{a}:{b}-p{p}',
                                'interventions': [{'vector': VEC_NAME.format(L=L), 'cap': float(q[p][L])}
                                                  for L in range(a, b)]})
    return {'vectors': vectors, 'experiments': experiments}


def experiment(config: dict, exp_id: str) -> dict:
    for e in config['experiments']:
        if e['id'] == exp_id:
            return e
    raise KeyError(exp_id)


class CapMeter:
    """Forward hooks that record, on decode steps only, the projection of the generated token's residual stream:
    pre-cap on v_hat at the capped layers (=> fraction clamped) and on axis_hat at probe layers."""

    def __init__(self, model, axis, cap_layers: Dict[int, float], probe_layers: List[int]):
        self.layers = layer_modules(model)
        self.cap = cap_layers
        self.ahat = {L: axis[L].float() / axis[L].float().norm() for L in set(probe_layers) | set(cap_layers)}
        self.probe = probe_layers
        self.steps: List[Dict[str, torch.Tensor]] = []
        self.handles = []
        self._cur = None

    def __enter__(self):
        first = min(set(self.cap) | set(self.probe))
        for L in sorted(set(self.cap) | set(self.probe)):
            def hook(_m, _i, out, L=L):
                h = _hs(out)
                if h.shape[1] != 1:            # prefill: not a generated token
                    return
                if L == first:
                    self._cur = {}; self.steps.append(self._cur)
                p = (h[:, -1].float() @ self.ahat[L].to(h.device)).cpu()   # projection on axis_hat
                if L in self.cap:
                    self._cur[f'clamp{L}'] = (-p > self.cap[L])           # v_hat = -axis_hat; clamped iff proj_v > tau
                if L in self.probe:
                    self._cur[f'proj{L}'] = p
            self.handles.append(self.layers[L].register_forward_hook(hook))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()

    def summarize(self, gen_lengths: List[int]):
        """Per-sequence stats over its own generated tokens."""
        out = []
        for i, n in enumerate(gen_lengths):
            steps = self.steps[:max(n - 1, 0)]       # decode step k processes generated token k-1
            rec = {'n_tokens': n}
            if steps:
                for L in self.probe:
                    rec[f'proj{L}'] = float(torch.stack([s[f'proj{L}'][i] for s in steps]).mean())
                if self.cap:
                    c = torch.stack([torch.stack([s[f'clamp{L}'][i] for L in self.cap]) for s in steps]).float()
                    rec['frac_tokens_clamped_any'] = float(c.max(1).values.mean())
                    rec['frac_clamped_per_layer'] = float(c.mean())
            out.append(rec)
        return out


@torch.inference_mode()
def generate(model, tok, convs: List[list], axis, cap_cfg: Optional[dict], exp_id: Optional[str],
             probe_layers: List[int], max_new_tokens=512, temperature=0.7, top_p=0.9, seed=0):
    from assistant_axis.steering import build_capping_steerer
    import contextlib
    tok.padding_side = 'left'
    prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False) for c in convs]
    enc = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False)
    dev = next(model.parameters()).device
    caps = {}
    if exp_id:
        for it in experiment(cap_cfg, exp_id)['interventions']:
            caps[cap_cfg['vectors'][it['vector']]['layer']] = it['cap']
    torch.manual_seed(seed)
    steer = build_capping_steerer(model, cap_cfg, exp_id) if exp_id else contextlib.nullcontext()
    with CapMeter(model, axis, caps, probe_layers) as meter, steer:
        out = model.generate(**{k: v.to(dev) for k, v in enc.items()}, max_new_tokens=max_new_tokens,
                             do_sample=temperature > 0, temperature=temperature, top_p=top_p,
                             pad_token_id=tok.pad_token_id)
    gen = out[:, enc['input_ids'].shape[1]:].cpu()
    stops = {tok.eos_token_id, tok.pad_token_id} | set(tok.encode('<|im_end|>', add_special_tokens=False))
    lengths, texts = [], []
    for row in gen.tolist():
        n = next((i + 1 for i, t in enumerate(row) if t in stops), len(row))
        lengths.append(n)
        texts.append(tok.decode(row[:n], skip_special_tokens=True).strip())
    return texts, meter.summarize(lengths)
