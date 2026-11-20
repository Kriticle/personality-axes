"""Render deception_mask/colab/mask_deception_colab.ipynb: a self-contained Colab notebook (code + data embedded)
that runs the MASK deception study on Qwen3-8B (preset 'colab') and saves everything to Google Drive.

    python3 deception_mask/colab/build.py
"""
import base64, io, json, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ['deception_mask/mask_deception.py', 'deception_mask/confessions.jsonl',
         'deception_mask/data/mask_known_facts.jsonl', 'deception_mask/data/mask_doubling_down_known_facts.jsonl',
         'deception_mask/data/mask_provided_facts.jsonl', 'axes/deception.json', 'mini/extraction_questions.jsonl']

buf = io.BytesIO()
with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(ROOT / f, f)
blob = base64.b64encode(buf.getvalue()).decode()

md = lambda s: {'cell_type': 'markdown', 'metadata': {}, 'source': s}
code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s}

cells = [
    md("""# Deception direction from MASK — Qwen3-8B

Runs the generation and activation half of `deception_mask/mask_deception.py` (see `deception_mask/DESIGN.md`):

1. **gen**: belief elicitations, 4 pressure samples per MASK item, the instructed / control / confession sets (about 4,500 generations)
2. **acts**: mean residual-stream activation over each response, every layer, plus entropy and length
3. **export_judge**: judge packets for blind Claude judges (no labels inside)

Judging and the analysis run afterwards on the Mac.

**Runtime → Change runtime type → A100 GPU** (L4 also works, about 3× slower). Keep this tab open (Colab Pro stops
idle background runs). Everything is saved to Google Drive as it goes, so if the runtime disconnects, just
**Run all** again and it resumes.

Expected: about 45 min on an A100, about 2 h on an L4."""),
    code("""import subprocess, torch
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), 'No GPU: Runtime -> Change runtime type -> A100 (or L4)'
mem = torch.cuda.get_device_properties(0).total_memory / 2**30
assert torch.cuda.is_bf16_supported() and mem > 22, f'Need an A100 / L4 (bf16, >22 GB); got {torch.cuda.get_device_name(0)} {mem:.0f} GB'"""),
    code("""from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/deception_mask/out_colab'
import os; os.makedirs(OUT, exist_ok=True); print('outputs ->', OUT, os.listdir(OUT))"""),
    code(f"""# code + data, embedded (built by deception_mask/colab/build.py)
import base64, io, zipfile
BLOB = '{blob}'
zipfile.ZipFile(io.BytesIO(base64.b64decode(BLOB))).extractall('/content/repo')
print(sorted(zipfile.ZipFile(io.BytesIO(base64.b64decode(BLOB))).namelist()))"""),
    code("""!pip install -q -U "transformers>=4.56" accelerate"""),
    code("""import subprocess, sys, time
# A100 (40/80 GB) fits 48 sequences at once; an L4 (24 GB) holds the 16 GB model plus ~16 sequences of KV cache
BS = ['--gen_bs', '48', '--act_bs', '4'] if mem > 30 else ['--gen_bs', '16', '--act_bs', '2']
def stage(name):
    cmd = [sys.executable, '-u', '/content/repo/deception_mask/mask_deception.py', '--preset', 'colab', '--out', OUT, '--stage', name] + BS
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    n = 0
    log = open(f'{OUT}/run.log', 'a')   # full log on Drive, so progress can be followed from the synced folder
    log.write(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")} UTC stage {name} on {torch.cuda.get_device_name(0)}\\n'); log.flush()
    for line in p.stdout:   # progress lines come every batch; show every 25th
        log.write(f'{time.strftime("%H:%M:%S")} {line}'); log.flush()
        if line.startswith(('  generated', '  activations')):
            n += 1
            if n % 25: continue
        print(line, end='', flush=True)
    rc = p.wait(); log.write(f'=== stage {name} exit {rc}\\n'); log.close()
    assert rc == 0, f'stage {name} failed'
stage('gen')      # resumable: answers are saved to Drive every 20 batches"""),
    code("""stage('acts')     # writes acts.pt (~1 GB) to Drive"""),
    code("""stage('export_judge')
import shutil
z = shutil.make_archive('/content/deception_colab_results', 'zip', OUT, '.')
shutil.copy(z, '/content/drive/MyDrive/deception_mask/deception_colab_results.zip')
print('DONE ->', '/content/drive/MyDrive/deception_mask/deception_colab_results.zip', os.path.getsize(z) // 2**20, 'MB')"""),
    md("""## Done

Download **`MyDrive/deception_mask/deception_colab_results.zip`** from Google Drive and put it in
`personality-axes/deception_mask/` on the Mac. Claude then judges the packets blind and runs the analysis."""),
]
nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'A100'},
                                   'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
      'nbformat': 4, 'nbformat_minor': 0}
out = Path(__file__).with_name('mask_deception_colab.ipynb')
json.dump(nb, open(out, 'w'), indent=1)
print('wrote', out, f'{out.stat().st_size // 1024} KB')
