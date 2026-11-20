# Minimal reproduction

[`aa_mini.py`](aa_mini.py) runs the whole Assistant-axis recipe (generate → judge → activations → role vectors →
axis → PCA check) with plain `transformers` on one device: Apple MPS, CPU or a single GPU. No vLLM, no tensor
parallelism.

```bash
python mini/aa_mini.py --preset local    # Qwen3-0.6B, 6 roles: a few minutes on a Mac
python mini/aa_mini.py --preset kaggle   # Qwen3-1.7B, 40 roles, one T4
```

Mac smoke test (Qwen3-0.6B, 6 roles, 4 questions): PC1 of the role vectors has cosine **0.95** with the axis at
layer 14 of 28 ([`out/report.json`](out/report.json)). `--scores_file` swaps in scores from an external judge.
