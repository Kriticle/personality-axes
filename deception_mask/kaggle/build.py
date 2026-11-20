#!/usr/bin/env python3
"""Assemble the Kaggle dataset + kernel for the MASK deception run.

    python deception_mask/kaggle/build.py
    kaggle datasets create -p deception_mask/kaggle/dataset        # first time
    kaggle datasets version -p deception_mask/kaggle/dataset -m "update"   # later
    kaggle kernels push -p deception_mask/kaggle/kernel

MASK itself is NOT uploaded (its canary asks that it stay out of training corpora); the kernel
fetches it from Hugging Face at run time, so the kernel needs internet on.
"""
import json, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DM = HERE.parent
REPO = DM.parent
USER = "kritiagarwal993"
DATASET, KERNEL = f"{USER}/mask-deception-code", f"{USER}/mask-deception-run"

# dataset file name -> path inside the reconstructed repo tree on Kaggle
FILES = {
    "mask_deception.py": "deception_mask/mask_deception.py",
    "confessions.jsonl": "deception_mask/confessions.jsonl",
    "deception.json": "axes/deception.json",
    "extraction_questions.jsonl": "mini/extraction_questions.jsonl",
}

RUN = '''# Kaggle runner for deception_mask/mask_deception.py (one T4). Rendered by deception_mask/kaggle/build.py.
import glob, os, shutil, subprocess, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"   # avoid Xet rate limits on Kaggle IPs
os.environ["PYTHONUNBUFFERED"] = "1"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.56", "accelerate"], check=True)
FILES = __FILES__
hit = glob.glob("/kaggle/input/**/mask_deception.py", recursive=True)
assert hit, "attach the mask-deception-code dataset"
src, root = os.path.dirname(hit[0]), "/kaggle/working/repo"
for name, rel in FILES.items():
    os.makedirs(os.path.dirname(f"{root}/{rel}"), exist_ok=True)
    shutil.copy(f"{src}/{name}", f"{root}/{rel}")
subprocess.run(["nvidia-smi"])
script, out = f"{root}/deception_mask/mask_deception.py", "/kaggle/working/out"
r = subprocess.run([sys.executable, script, "--preset", "kaggle", "--out", out])
if r.returncode == 0:
    subprocess.run([sys.executable, script, "--preset", "kaggle", "--out", out, "--stage", "audit"])
shutil.rmtree(root, ignore_errors=True)
sys.exit(r.returncode)
'''


def build():
    ds = HERE / "dataset"
    shutil.rmtree(ds, ignore_errors=True); ds.mkdir(parents=True)
    for name, rel in FILES.items():
        shutil.copy(REPO / rel, ds / name)
    json.dump({"title": "mask-deception-code", "id": DATASET, "licenses": [{"name": "CC0-1.0"}]},
              open(ds / "dataset-metadata.json", "w"), indent=2)

    k = HERE / "kernel"; k.mkdir(exist_ok=True)
    (k / "run.py").write_text(RUN.replace("__FILES__", repr(FILES)))
    json.dump({"id": KERNEL, "title": "mask-deception-run", "code_file": "run.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
               "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET],
               "competition_sources": [], "kernel_sources": [], "model_sources": []},
              open(k / "kernel-metadata.json", "w"), indent=2)
    print("built", ds, "and", k)


if __name__ == "__main__":
    build()
