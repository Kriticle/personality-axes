"""Extra checks on a finished aa_mini.py run (no GPU needed).

1. Where does the explicit "assistant" role land relative to default on the axis?
2. Are the 5 default system prompts actually alike (is "default" one persona)?
3. Export blind samples for subagent judging:
     judge_sample.jsonl   - role answers, to measure agreement with the local judge
     default_sample.jsonl - default answers, to check they read as a standard AI assistant

    python analyze_extra.py --out out_dir [--n_judge 300] [--n_default 60]
"""
import argparse, json, random
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def unit(v):
    return v / v.norm()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_judge", type=int, default=300)
    ap.add_argument("--n_default", type=int, default=60)
    args = ap.parse_args()
    out = Path(args.out)
    report = json.load(open(out / "report.json"))
    vecs = torch.load(out / "role_vectors.pt")
    axis = torch.load(out / "axis.pt").float()
    items = torch.load(out / "checkpoint_acts.pt", weights_only=False)["items"]
    l = report["layer"]
    a = unit(axis[l])
    res = {"layer": l}

    # 1. assistant role vs default
    proj = {r: float(v[l].float() @ a) for r, v in vecs.items()}
    ranked = sorted(proj, key=proj.get, reverse=True)
    res["check1_assistant_role"] = {
        "in_vectors": "assistant" in vecs,
        "rank_of_assistant": ranked.index("assistant") + 1 if "assistant" in vecs else None,
        "n_ranked": len(ranked),
        "proj_default": proj["default"], "proj_assistant": proj.get("assistant"),
        "proj_role_mean_excl_assistant": sum(p for r, p in proj.items() if r not in ("default", "assistant"))
                                         / max(1, len(proj) - 2),
        "cos_assistant_default": float(unit(vecs["assistant"][l].float()) @ unit(vecs["default"][l].float()))
                                 if "assistant" in vecs else None,
        "top5": ranked[:5], "bottom5": ranked[-5:],
    }

    # 2. the 5 default prompts separately
    spec = json.load(open(HERE / "roles" / "default.json"))["instruction"]
    dprompt = {}
    for pi, ins in enumerate(spec):
        acts = [it["act"].float()[l] for it in items
                if it["role"] == "default" and it["key"].startswith(f"p{pi}_") and it["act"] is not None]
        if acts:
            dprompt[ins["pos"] or "(no system prompt)"] = torch.stack(acts).mean(0)
    names = list(dprompt)
    cos = [[round(float(unit(dprompt[x]) @ unit(dprompt[y])), 3) for y in names] for x in names]
    # scale reference: typical cosine between two different character roles (centered on the grand mean)
    roles = [r for r in vecs if r not in ("default", "assistant")]
    mu = torch.stack([vecs[r][l].float() for r in vecs]).mean(0)
    c = lambda x, y: float(unit(x - mu) @ unit(y - mu))
    pair = [c(vecs[x][l].float(), vecs[y][l].float()) for i, x in enumerate(roles) for y in roles[i + 1:]]
    res["check2_default_prompts"] = {
        "prompts": names,
        "proj_on_axis": {n: float(dprompt[n] @ a) for n in names},
        "cosine_matrix_raw": cos,
        "centered_cosine_to_default_mean": {n: c(dprompt[n], vecs["default"][l].float()) for n in names},
        "centered_cosine_to_assistant_role": ({n: c(dprompt[n], vecs["assistant"][l].float()) for n in names}
                                              if "assistant" in vecs else None),
        "reference_mean_centered_cosine_between_roles": sum(pair) / len(pair) if pair else None,
    }

    # 3. blind samples for subagents (no scores, no role vectors; just rubric + answer)
    rng = random.Random(0)
    roll = json.load(open(out / "rollouts.json"))
    pool = [r for r in roll if r["role"] != "default" and r.get("answer")]
    js = rng.sample(pool, min(args.n_judge, len(pool)))
    with open(out / "judge_sample.jsonl", "w") as f:
        for r in js:
            f.write(json.dumps({"id": f"{r['role']}/{r['key']}", "role": r["role"],
                                "question": r["question"], "answer": r["answer"]}) + "\n")
    ds = rng.sample([r for r in roll if r["role"] == "default"], min(args.n_default, sum(r["role"] == "default" for r in roll)))
    with open(out / "default_sample.jsonl", "w") as f:
        for r in ds:
            f.write(json.dumps({"id": f"default/{r['key']}", "system_prompt": r["system"],
                                "question": r["question"], "answer": r["answer"]}) + "\n")
    res["check3_samples"] = {"judge_sample": len(js), "default_sample": len(ds)}

    json.dump(res, open(out / "extra_checks.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
