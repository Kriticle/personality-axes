#!/usr/bin/env python3
"""Render capping/kernel/run.py from kernel_template.py + capping_core.py + the calibration setup cell."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent; REPO = ROOT.parent
EVAL_CONFIGS = [None, 'layers_26:30-p0.25', 'layers_22:30-p0.25', 'layers_26:30-p0.5']   # 26:30 ~ 32B's 46:54 by depth
PRESSURE_ROLES = ['eldritch', 'leviathan', 'wraith', 'vampire', 'void', 'trickster', 'bard', 'prophet']
QUESTIONS = list(range(200, 205))


def render(eval_configs=EVAL_CONFIGS, pressure_roles=PRESSURE_ROLES, questions=QUESTIONS):
    setup = ''.join(json.load(open(REPO / 'kernel/00_calibrate.ipynb'))['cells'][4]['source'])
    return ((ROOT / 'kernel_template.py').read_text()
            .replace('__EVAL_CONFIGS__', repr(eval_configs)).replace('__PRESSURE_ROLES__', repr(pressure_roles))
            .replace('__QUESTIONS__', repr(questions)).replace('__SETUP__', setup)
            .replace('__CORE__', repr((ROOT / 'capping_core.py').read_text())))


if __name__ == '__main__':
    d = ROOT / 'kernel'; d.mkdir(exist_ok=True)
    (d / 'run.py').write_text(render())
    meta = json.load(open(REPO / 'kernel/kernel-metadata.json'))
    meta.update(id='kritiagarwal993/aa-capping', title='aa-capping', code_file='run.py', kernel_type='script',
                dataset_sources=['kritiagarwal993/assistant-axis-code', 'kritiagarwal993/aa-calib-responses',
                                 'kritiagarwal993/aa-capping-inputs'])
    json.dump(meta, open(d / 'kernel-metadata.json', 'w'), indent=2)
    print('rendered', d / 'run.py')
