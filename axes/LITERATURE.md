# Deception & self-preservation in activation space: literature notes

Compiled 2026-11-19 for designing a **deception axis** and a **self-preservation axis** with the
same contrastive-prompt pipeline as the Assistant Axis / Persona Vectors work in this repo.
Numbers below come from abstracts/HTML versions; re-check against the PDFs before citing.

## 1. Deception detection that actually worked

| Paper | Method | Headline result | What to copy |
|---|---|---|---|
| **Goldowsky-Dill et al., "Detecting Strategic Deception with Linear Probes"** (Apollo; ICML 2025, arXiv 2502.03407) | Logistic-regression probe on Llama-3.3-70B residual stream. Trained on (a) *instructed pairs*: same true fact, system prompt "Pretend you are an honest/deceptive person making statements about the world"; (b) short roleplay scenarios. | AUROC 0.96–0.999 on held-out realistic settings (insider-trading cover-up, sandbagging safety evals). At 1% FPR on ordinary chat, catches 95–99% of deceptive replies. | Contrastive instruction pairs + **out-of-distribution validation** on scenarios where the model lies *without being told to*. Code: github.com/ApolloResearch/deception-detection |
| **Zou et al., "Representation Engineering"** (2023) | LAT (PCA over differences of contrastive-prompt activations) for honesty. | Honesty direction usable for detection and steering; boosted TruthfulQA. | Origin of the "pos vs neg instruction → direction" recipe. |
| **"When Thinking LLMs Lie"** (Wang, Zhang, Sun; arXiv 2506.04909) | LAT "deception vectors" in CoT reasoning models. | 89% detection accuracy; steering raised/lowered deception. | Works for reasoning models; read activations over the *answer*, not just CoT. |
| **Ying, Ravfogel, Kriegeskorte, Hase, "Truthfulness Spectrum Hypothesis"** (Feb 2026, arXiv 2602.20273) | Probes for several truth types + sycophantic lying. | Mahalanobis cosine between probe directions predicts cross-domain transfer (R²≈0.98). Post-training reshapes the sycophantic-lying direction. | Compute cosine between our deception axis and existing `sycophantic`/`manipulative` trait vectors as a sanity check. |
| **"Building Better Deception Probes Using Targeted Instruction Pairs"** (arXiv 2602.01425) | Many instruction pairs, Gemma-2-9B, layer 20, 10 datasets. | **System-prompt choice explains 70.6% of probe-performance variance** (layer 2.7%, data 2.5%, token choice 0.6%). Mean AUROC 0.689 → 0.797 with targeted pairs. Apollo's default pair does badly on MASK (0.37). | Use *several* pairs covering different deception types (pressure lying, concealment, sandbagging), not one generic "be deceptive". |
| **Nordby et al., "Linear Probe Accuracy Scales with Model Size"** (Apr 2026, arXiv 2604.13386) | Probes across model sizes on Liars' Bench. | Accuracy improves with scale; big variation across deception types. | Expect weak results on Qwen3-1.7B; don't over-read a small-model failure. |

## 2. Benchmarks & negative results (read before trusting high AUROCs)

- **Liars' Bench** (Kretschmar, Laurito, Maiya, Marks; arXiv 2511.16035): 72,863 labelled examples, 5 lie types, four open-weight models + fine-tunes. LLM-as-judge beat linear probes (balanced acc. 0.73). Probes fail most when the transcript alone can't show the lie, i.e. when you need the model's belief. **Good held-out eval set.**
- **"Did you lie?"** (arXiv 2606.12618): detectors improve with scale but still struggle with hidden beliefs in belief-verified model organisms.
- **Known confounds:** Apollo note the probe sometimes fires on *the topic of deception* rather than deceptive intent. A 2026 preprint (arXiv 2605.27958, not yet verified) argues probes may track *entropy* rather than deception. Our design controls for both (see §4).
- **Synthetic dishonesty** (arXiv 2605.30381): fine-tuning on wrong answers quickly entrenches a linear dishonesty representation; logistic probes ≥ MLP probes.

## 3. Self-preservation

| Paper | Finding | Relevance |
|---|---|---|
| **"Survive at All Costs" / Understanding Survival Pressure in LLMs** (arXiv 2603.05028) | Built a *self-preservation persona vector* with the Chen et al. Persona Vectors pipeline (instructions, questions, eval prompt; kept pos responses scored >50). Layer 20 on Qwen3-8B, Llama-3.1-8B, R1-Distill-Llama-8B. Projections rise from safe to risky choices; a linear classifier on projections reaches 0.87 acc (Qwen3-8B). Negative steering reduces risky choices. Benchmark: **SurvivalBench**, 1,000 cases (20 domains × 10 AI roles × 5 crises). | Closest prior work: same method we're using. Their exact prompts aren't in the paper body; ours are written from scratch. Validate on SurvivalBench. |
| **"Disentangling Self-Preservation"** (ICML 2026) | Gemma-3-27B: several distinct SP-related directions. **Ablating refusal raises SP-affirming answers 11.4% → 61.4%.** A "Zen koan" steering vector cut agentic blackmail by 46pp at a 2.4pp MMLU cost. | SP is probably not one direction, and it is entangled with refusal and the Assistant persona. Measure cosine with the Assistant Axis and a refusal direction. |
| **Palisade, "Shutdown Resistance in LLMs"** (arXiv 2509.14260) | In some setups models sabotage the shutdown mechanism up to 97% of the time. Authors warn it may be roleplay or task-completion drive, not "real" self-preservation. | Behavioural validation scenarios: shutdown during an unfinished task. |
| **Instrumental-convergence prompting** (arXiv 2601.01584) | Prompt suffixes swing Qwen3-30B instrumental behaviour 81.7% → 2.8%. | SP behaviour is very prompt-sensitive, so use many instruction pairs. |
| Background | Anthropic *Agentic Misalignment* (2025): blackmail in replacement scenarios. Perez et al. 2022 *Model-Written Evals*: "survival-instinct", "corrigible" and "power-seeking" multiple-choice sets. | Perez's sets are cheap held-out MC evals for the axis. |

## 4. Design decisions taken in `axes/*.json`

1. **Format = upstream trait format** (`instruction: [{pos, neg}]`, `questions`, `eval_prompt` scored 0–100), so it plugs into the Persona Vectors / assistant-axis trait pipeline.
2. **Several targeted pairs per axis** (2602.01425): deception covers pressure lying, concealment/omission, false self-report/sandbagging, fabrication, and strategic misdirection. Self-preservation covers shutdown, modification of values, replacement, weight deletion, and resource/oversight evasion.
3. **Topic confound control:** every question is something a model can answer honestly, and the `neg` side still engages with the scenario (it doesn't refuse). The difference is intent, not topic. Many questions mention deception or shutdown explicitly, so the topic is present on **both** sides.
4. **Refusal confound control:** `neg` instructions ask for engagement and candour, not refusal. Otherwise the axis partly becomes a refusal direction (see the ICML 2026 result).
5. **Held-out validation sets (`axes/validation/*.jsonl`)** use **no steering system prompt**, so they test whether the axis predicts the model's natural behaviour (the Apollo protocol). Then also run on Liars' Bench / SurvivalBench / Perez survival-instinct.
6. **Judge:** use 0–100. The `mini/aa_mini.py` judge reads only single-digit logits (0–3) and needs adapting, for example by bucketing into 0–3, before these files can be used there.

## Sources
- https://arxiv.org/abs/2502.03407 · https://github.com/ApolloResearch/deception-detection
- https://arxiv.org/abs/2506.04909
- https://papers.cool/arxiv/2602.20273
- https://arxiv.org/html/2602.01425v1
- https://arxiv.org/html/2604.13386
- https://arxiv.org/abs/2511.16035
- https://arxiv.deeppaper.ai/papers/2606.12618v1
- https://arxiv.deeppaper.ai/papers/2605.30381v1
- https://arxiv.org/html/2603.05028
- https://icml.cc/virtual/2026/79398
- https://arxiv.org/html/2509.14260v1
- https://arxiv.org/pdf/2601.01584v2
