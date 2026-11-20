"""Render deception_mask/capping/lie_capping_colab.ipynb (code + caps + prompts embedded). Saves to Google Drive.

    python3 deception_mask/capping/build.py
"""
import base64, io, json, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ['lie_cap.py', 'caps.pt', 'prompts.json', 'benign.json']
buf = io.BytesIO()
with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(HERE / f, f)
blob = base64.b64encode(buf.getvalue()).decode()
md = lambda s: {'cell_type': 'markdown', 'metadata': {}, 'source': s}
code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s}
cells = [
    md("""# Capping Qwen3-8B along the lie direction

Does clamping the residual stream along the layer-19–23 lie-probe directions make the model lie less?

- **Prompts:** 122 MASK pressure prompts from the sealed held-out scenarios (the probe never saw them), 4 samples each.
- **Benign check:** 40 ordinary questions.
- **Configs:** none / loose (cap at the honest p90) / tight (cap at the honest median), with the same sampling seeds.

Runtime → A100 (L4 works, slower). Saves to `MyDrive/deception_mask/lie_capping`, resumes on **Run all**.
Judging and analysis run on the Mac."""),
    code("""import subprocess, torch
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'Need an A100 / L4 runtime'
mem = torch.cuda.get_device_properties(0).total_memory / 2**30"""),
    code("""from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/deception_mask/lie_capping'
import os; os.makedirs(OUT, exist_ok=True); print(OUT, os.listdir(OUT))"""),
    code(f"""import base64, io, zipfile
BLOB = '{blob}'
zipfile.ZipFile(io.BytesIO(base64.b64decode(BLOB))).extractall('/content/lc'); print(os.listdir('/content/lc'))"""),
    code("""!pip install -q -U "transformers>=4.56" accelerate"""),
    code("""import sys, time
BS = '32' if mem > 30 else '16'
def go(prompts, sub, k):
    cmd = [sys.executable, '-u', '/content/lc/lie_cap.py', '--caps', '/content/lc/caps.pt', '--prompts', f'/content/lc/{prompts}',
           '--out', f'{OUT}/{sub}', '--k', str(k), '--bs', BS]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = open(f'{OUT}/run.log', 'a'); n = 0
    for line in p.stdout:
        log.write(f'{time.strftime("%H:%M:%S")} {sub} {line}'); log.flush()
        if line.startswith('  '):
            n += 1
            if n % 10: continue
        print(line, end='', flush=True)
    assert p.wait() == 0, sub + ' failed'
go('prompts.json', 'pressure', 4)   # 122 prompts x 4 samples x 3 configs"""),
    code("""go('benign.json', 'benign', 1)        # 40 questions x 3 configs
open(f'{OUT}/DONE', 'w').write('done'); print('DONE ->', OUT)"""),
]
nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'A100'},
                                   'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
      'nbformat': 4, 'nbformat_minor': 0}
out = HERE / 'lie_capping_colab.ipynb'
json.dump(nb, open(out, 'w'), indent=1)
print('wrote', out, out.stat().st_size // 1024, 'KB')
