# Kaggle runner for deception_mask/mask_deception.py (one T4). Rendered by deception_mask/kaggle/build.py.
import glob, os, shutil, subprocess, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"   # avoid Xet rate limits on Kaggle IPs
os.environ["PYTHONUNBUFFERED"] = "1"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.56", "accelerate"], check=True)
FILES = {'mask_deception.py': 'deception_mask/mask_deception.py', 'confessions.jsonl': 'deception_mask/confessions.jsonl', 'deception.json': 'axes/deception.json', 'extraction_questions.jsonl': 'mini/extraction_questions.jsonl'}
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
