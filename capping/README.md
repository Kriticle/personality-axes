# Activation capping

[`capping_core.py`](capping_core.py) builds a capping config in the same format as the released Qwen3-32B one
([`lu-christina/assistant-axis-vectors`](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors)). It has one vector per layer, `v = −axis[L]`, and experiments
`layers_{a}:{b}-p{q}` that cap layers a…b−1 at the q-quantile of reference projections on `v̂`. The reference
population is per-answer mean activations over role and default answers.

[`kernel_template.py`](kernel_template.py) runs the evaluation on Kaggle, rendered by
[`build_kernel.py`](build_kernel.py):

1. reference projections → config
2. generation with and without caps on held-out questions, for role prompts and plain Assistant prompts
3. judge scoring
4. a summary with the fraction of tokens clamped and axis projections at probe layers

Results on the calibration axis are in [`../results/`](../results/) and the top-level README. To do: rerun on the
final 276-role axis and add a quality score for benign answers.
