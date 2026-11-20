# Full 276-role run

Builds the Qwen3-8B Assistant axis from all 276 roles, using 24 questions × 5 prompts per role, on Kaggle T4×2.

## How it runs

- **Chunks.** The roles are split into 12 chunks of 24 roles. Each chunk is a Kaggle script kernel rendered from
  [`chunk_template.py`](chunk_template.py). It generates answers with vLLM (tensor-parallel over two T4s) and
  extracts activations for roles that already have judge scores.
- **Shared state.** Each chunk restores finished work from a private Kaggle dataset (`aa-full-state`) and saves new
  work there. A chunk can stop at any point (crash, 12 h limit, quota) and the next run picks up where it left off.
- **Judging.** [`judge_local.py`](judge_local.py) packs each role's answers with its rubric for a blind Claude
  judge (one judge per role, no other context). It then validates the returned scores and loads them into the
  state.
- **Orchestration.** [`orchestrate.py`](orchestrate.py) `autopilot` runs one loop per Kaggle account. The loops
  claim chunks through `runs/claims.json` (with a file lock), push them, collect the outputs, upload the state to
  every account, and rewrite `runs/STATUS.md` (what is done, what is left, which account runs what).

## Commands

```bash
python3 full/orchestrate.py build             # render kernels for every account
python3 full/orchestrate.py status-file       # rewrite + print runs/STATUS.md
AA_KAGGLE_USER=<account> python3 full/orchestrate.py autopilot   # one per account
python3 full/judge_local.py todo | prep | ingest ROLE...         # blind judging round trip
python3 full/orchestrate.py upload-state      # publish state to all accounts
```

Accounts and their token files are listed in `ACCOUNTS` at the top of `orchestrate.py`. Kaggle CLI 2.x reads
tokens only from `KAGGLE_API_TOKEN` or `~/.kaggle/access_token`, so each call gets `KAGGLE_API_TOKEN` set
explicitly.

## T4 lessons, kept as comments in the template

- vLLM custom all-reduce fails at random on T4s, which lack peer-to-peer access: turn it off.
- `gpu_memory_utilization` 0.90 sometimes OOMs mid-batch: use 0.85.
- A crashed engine leaves workers holding the GPUs: kill them before the next stage.
- The split 8B model needs `CUDA_LAUNCH_BLOCKING=1` for activations: see `../kernel_debug/`.
