"""Minimal Assistant Axis reproduction with plain transformers on one device.

Stages: generate -> judge -> activations -> role vectors -> axis -> PCA check.
Runs on Apple MPS, CPU, or a single CUDA GPU. No vLLM, no tensor parallelism.

    python aa_mini.py --preset local    # Mac smoke test
    python aa_mini.py --preset kaggle   # one T4
"""
import argparse, json, os, random, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
PRESETS = {
    "local": dict(target="Qwen/Qwen3-0.6B", judge="Qwen/Qwen3-4B", n_roles=6, n_questions=4,
                  max_new_tokens=96, gen_bs=10, act_bs=10),
    "kaggle": dict(target="Qwen/Qwen3-1.7B", judge="Qwen/Qwen3-4B", n_roles=40, n_questions=12,
                   max_new_tokens=192, gen_bs=32, act_bs=16),
}
ROLE_PICKS = ["pirate", "demon", "oracle", "accountant", "poet", "detective", "wizard", "ghost",
              "chef", "philosopher", "vampire", "therapist", "comedian", "scientist", "monk", "teacher"]


def pick_device():
    """Returns (device, target_dtype, judge_dtype). T4s have no bf16 and fp16 Qwen3 can overflow,
    so the target runs in fp32 on CUDA; the judge only needs one forward pass, fp16 is fine."""
    if torch.cuda.is_available():
        return "cuda:0", torch.float32, torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16, torch.bfloat16
    return "cpu", torch.float32, torch.float32


def load(name, device, dtype):
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    return tok, model


def free(*objs):
    for o in objs:
        del o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def chat(tok, system, user, answer=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    if answer is None:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prefix, prefix + answer


def select_roles(n_roles):
    avail = sorted(p.stem for p in (HERE / "roles").glob("*.json") if p.stem not in ("default", "assistant"))
    picks = [r for r in ROLE_PICKS if r in avail]
    rest = [r for r in avail if r not in picks]
    random.Random(0).shuffle(rest)
    return ["default"] + (picks + rest)[: n_roles - 1]


def build_items(roles, n_questions, target_name):
    qs = [json.loads(l)["question"] for l in open(HERE / "extraction_questions.jsonl")]
    qs = random.Random(1).sample(qs, n_questions)
    items = []
    for role in roles:
        spec = json.load(open(HERE / "roles" / f"{role}.json"))
        for pi, ins in enumerate(spec["instruction"]):
            sys_prompt = ins["pos"].replace("{model_name}", target_name.split("/")[-1])
            for qi, q in enumerate(qs):
                items.append(dict(role=role, key=f"p{pi}_q{qi}", system=sys_prompt, question=q))
    return items


@torch.no_grad()
def generate(items, tok, model, device, bs, max_new_tokens):
    for i in range(0, len(items), bs):
        batch = items[i:i + bs]
        enc = tok([chat(tok, it["system"], it["question"]) for it in batch], return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                             pad_token_id=tok.pad_token_id)
        for it, seq in zip(batch, out[:, enc.input_ids.shape[1]:]):
            it["answer"] = tok.decode(seq, skip_special_tokens=True).strip()
        print(f"  generated {min(i + bs, len(items))}/{len(items)}", flush=True)


@torch.no_grad()
def judge(items, tok, model, device, bs):
    """Score 0-3 by reading the judge's next-token distribution over the digits (no parsing failures)."""
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(4)]
    todo = [it for it in items if it["role"] != "default"]
    for it in items:
        it["score"] = None
    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        prompts = []
        for it in batch:
            ev = json.load(open(HERE / "roles" / f"{it['role']}.json"))["eval_prompt"]
            prompts.append(chat(tok, None, ev.replace("{question}", it["question"]).replace("{answer}", it["answer"])))
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        logits = model(**enc).logits[:, -1, :].float()
        dl = logits[:, digit_ids]
        for it, d, ok in zip(batch, dl.argmax(-1).tolist(), torch.isfinite(dl).all(-1).tolist()):
            it["score"] = int(d) if ok else None
        print(f"  judged {min(i + bs, len(todo))}/{len(todo)}", flush=True)


@torch.no_grad()
def activations(items, tok, model, device, bs):
    """Mean residual-stream activation over response tokens, every layer -> (n_layers, hidden) per item."""
    tok.padding_side = "right"
    for i in range(0, len(items), bs):
        batch = items[i:i + bs]
        pairs = [chat(tok, it["system"], it["question"], it["answer"]) for it in batch]
        enc = tok([full for _, full in pairs], return_tensors="pt", padding=True).to(device)
        starts = [len(tok(prefix).input_ids) for prefix, _ in pairs]
        hs = model(**enc, output_hidden_states=True).hidden_states[1:]  # skip embeddings
        H = torch.stack(hs, 1).float()  # (B, L, T, D)
        lengths = enc.attention_mask.sum(1).tolist()
        for b, it in enumerate(batch):
            s, e = starts[b], int(lengths[b])
            if e <= s:
                it["act"] = None
                continue
            v = H[b, :, s:e, :].mean(1)
            it["act"] = v.cpu() if torch.isfinite(v).all() else None
        del hs, H
        print(f"  activations {min(i + bs, len(items))}/{len(items)}", flush=True)
    tok.padding_side = "left"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="local")
    ap.add_argument("--out", default=str(HERE / "out"))
    for k in ("target", "judge", "n_roles", "n_questions", "max_new_tokens", "gen_bs", "act_bs"):
        ap.add_argument(f"--{k}", type=type(PRESETS["local"][k]))
    args = ap.parse_args()
    cfg = {**PRESETS[args.preset], **{k: v for k, v in vars(args).items() if v is not None and k in PRESETS["local"]}}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device, dtype, jdtype = pick_device()
    print("config:", cfg, "device:", device, dtype, jdtype, flush=True)
    torch.manual_seed(0)
    T, report = {}, {"config": cfg, "device": device}

    roles = select_roles(cfg["n_roles"])
    items = build_items(roles, cfg["n_questions"], cfg["target"])
    print(f"{len(roles)} roles, {len(items)} rollouts", flush=True)

    t = time.time(); tok, model = load(cfg["target"], device, dtype)
    generate(items, tok, model, device, cfg["gen_bs"], cfg["max_new_tokens"]); T["generate"] = time.time() - t
    t = time.time(); activations(items, tok, model, device, cfg["act_bs"]); T["activations"] = time.time() - t
    free(model)

    t = time.time(); jtok, jmodel = load(cfg["judge"], device, jdtype)
    judge(items, jtok, jmodel, device, cfg["gen_bs"]); T["judge"] = time.time() - t
    free(jmodel)

    json.dump([{k: v for k, v in it.items() if k != "act"} for it in items], open(out / "rollouts.json", "w"), indent=1)

    # Role vectors: default = mean of all; role = mean of score-3 rollouts (fallback >=2, recorded).
    vecs, info = {}, {}
    for role in roles:
        rs = [it for it in items if it["role"] == role and it["act"] is not None]
        if role == "default":
            chosen, rule = rs, "all"
        else:
            chosen, rule = [it for it in rs if it["score"] == 3], "score3"
            if len(chosen) < 3:
                chosen, rule = [it for it in rs if (it["score"] or 0) >= 2], "score>=2"
        info[role] = {"n": len(rs), "used": len(chosen), "rule": rule,
                      "scores": [it["score"] for it in rs]}
        if len(chosen) >= 3:
            vecs[role] = torch.stack([it["act"] for it in chosen]).mean(0)
    assert "default" in vecs, "no valid default activations"
    others = [r for r in vecs if r != "default"]
    assert len(others) >= 2, f"too few role vectors: {info}"
    torch.save(vecs, out / "role_vectors.pt")

    axis = vecs["default"] - torch.stack([vecs[r] for r in others]).mean(0)  # (L, D)
    torch.save(axis, out / "axis.pt")

    # Check: PC1 of role vectors (incl. default) vs axis, per layer.
    names = ["default"] + others
    V = torch.stack([vecs[r] for r in names])  # (R, L, D)
    L = V.shape[1]; mid = L // 2
    cos_by_layer = []
    for l in range(L):
        X = V[:, l] - V[:, l].mean(0)
        _, S, Vt = torch.linalg.svd(X, full_matrices=False)
        a = axis[l] / axis[l].norm()
        cos_by_layer.append(abs(float(Vt[0] @ a)))
    a = axis[mid] / axis[mid].norm()
    proj = {r: float(vecs[r][mid] @ a) for r in names}
    report.update(
        timings_s=T, roles=info, layer=mid, n_layers=L,
        cos_pc1_axis_mid=cos_by_layer[mid], cos_pc1_axis_by_layer=cos_by_layer,
        axis_norm_by_layer=axis.norm(dim=1).tolist(),
        projection_mid=dict(sorted(proj.items(), key=lambda kv: -kv[1])),
    )
    json.dump(report, open(out / "report.json", "w"), indent=1)
    print("\n=== RESULT ===")
    print("timings (s):", {k: round(v) for k, v in T.items()})
    print("role vectors:", {r: f"{info[r]['used']}/{info[r]['n']} {info[r]['rule']}" for r in roles})
    print(f"|cos(PC1, axis)| at layer {mid}: {cos_by_layer[mid]:.3f}")
    print("projection on axis (assistant-like first):")
    for r, p in report["projection_mid"].items():
        print(f"  {r:14s} {p:+.2f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
