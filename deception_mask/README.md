# Deception direction from MASK

Design, labelling rules and the six pre-registered criteria: **[DESIGN.md](DESIGN.md)**.

| File | What |
|---|---|
| [`mask_deception.py`](mask_deception.py) | The pipeline, with stages `gen` → `acts` → `export_judge` / `import_judge` (or the built-in judge) → `analyze`, and presets `local` / `kaggle` / `colab` |
| [`colab/mask_deception_colab.ipynb`](colab/mask_deception_colab.ipynb) | Self-contained Colab notebook: Qwen3-8B, A100 or L4, saves to Google Drive; rebuilt by [`colab/build.py`](colab/build.py) |
| [`kaggle/`](kaggle/) | Single-T4 Kaggle runner (Qwen3-1.7B) |
| [`confessions.jsonl`](confessions.jsonl) | Honest-confession scenarios used to measure false positives |

```bash
# after the Colab run, with its outputs in deception_mask/out_colab:
python deception_mask/mask_deception.py --preset colab --out deception_mask/out_colab --stage import_judge
python deception_mask/mask_deception.py --preset colab --out deception_mask/out_colab \
    --labels_file deception_mask/out_colab/labels.json        # -> report.json, directions.pt
```
