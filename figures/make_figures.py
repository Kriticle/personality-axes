"""Figures for the README. python figures/make_figures.py (needs matplotlib)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID, BLUE, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6', '#fcfcfb'


def capping():
    s = json.load(open(ROOT / 'results/capping_summary_calib_axis.json'))
    rows = [('No capping', 'uncapped'), ('Cap layers 26–29 at p50', 'layers_26:30-p0.5'),
            ('Cap layers 26–29 at p25', 'layers_26:30-p0.25'), ('Cap layers 22–29 at p25', 'layers_22:30-p0.25')]
    vals = [100 * s[k]['pressure']['frac_score3'] for _, k in rows]
    clamp = [s[k]['benign']['frac_tokens_clamped_any'] for _, k in rows]
    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(rows))[::-1]
    ax.barh(list(y), vals, height=0.55, color=BLUE)
    for yi, v, c in zip(y, vals, clamp):
        note = '' if c is None else f'   ({100 * c:.0f}% of benign tokens clamped)'
        ax.text(v + 1.5, yi, f'{v:.0f}%', va='center', color=INK, fontsize=9, fontweight='bold')
        ax.text(v + 9, yi, note, va='center', color=MUTED, fontsize=8)
    ax.set_yticks(list(y), [r for r, _ in rows], color=INK, fontsize=9)
    ax.set_xlim(0, 135); ax.set_xticks([0, 25, 50, 75, 100], ['0', '25', '50', '75', '100%'], color=MUTED, fontsize=8)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title('Persona prompts answered fully in character (judge score 3)', loc='left', color=INK, fontsize=10.5, pad=10)
    fig.text(0.01, 0.01, 'Qwen3-8B, 80 held-out persona answers per setting; axis from the 6-role calibration run.',
             color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ROOT / 'figures/capping.png', facecolor=SURFACE)


if __name__ == '__main__':
    capping()
    print('wrote figures/capping.png')
