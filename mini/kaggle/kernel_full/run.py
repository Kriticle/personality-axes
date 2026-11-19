# Kaggle runner for aa_mini.py: one T4, plain transformers. Outputs land in /kaggle/working/out.
import glob, os, shutil, subprocess, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"  # avoid Xet rate limits on Kaggle IPs
os.environ["PYTHONUNBUFFERED"] = "1"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.56", "accelerate"], check=True)
hit = glob.glob("/kaggle/input/**/aa_mini.py", recursive=True)
assert hit, "attach the aa-mini-code dataset"
src = os.path.dirname(hit[0]); dst = "/kaggle/working/aa_mini"
shutil.copytree(src, dst, dirs_exist_ok=True)
for z in glob.glob(f"{dst}/*.zip"):  # directories are uploaded zipped
    shutil.unpack_archive(z, dst); os.remove(z)
subprocess.run(["nvidia-smi"])
r = subprocess.run([sys.executable, f"{dst}/aa_mini.py", "--preset", "kaggle", "--out", "/kaggle/working/out"])
shutil.rmtree(dst, ignore_errors=True)
sys.exit(r.returncode)
