"""Figures for the README. python figures/make_figures.py (needs matplotlib)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID, BLUE, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6', '#fcfcfb'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']   # validated categorical order
ORANGE = CAT[1]


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0, colors=MUTED, labelsize=8)
    ax.set_axisbelow(True)


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


def persona_map():
    """PCA of the finished role vectors (layer 18), coloured by projection on the calibration axis."""
    m = json.load(open(ROOT / 'results/persona_map_partial.json'))
    R = sorted(m['roles'], key=lambda r: r['proj'])
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200, facecolor=SURFACE); style(ax)
    p = [r['proj'] for r in R]; lo, hi = min(p), max(p)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('b', ['#cde2fb', '#2a78d6', '#184f95'])
    ax.scatter([r['pc1'] for r in R], [r['pc2'] for r in R], s=46, c=p, cmap=cmap, vmin=lo, vmax=hi,
               edgecolors=SURFACE, linewidths=1.5, zorder=3)
    label = R[:5] + R[-5:]
    placed = []
    for r in sorted(label, key=lambda r: -r['pc2']):     # nudge labels down until they clear earlier ones
        x, y = r['pc1'], r['pc2']; ly = y
        while any(abs(ly - py) < 0.9 and abs(x - px) < 4 for px, py in placed):
            ly -= 0.9
        placed.append((x, ly))
        ax.annotate(r['role'], (x, y), xytext=(x + 0.5, ly + 0.25), fontsize=7.5, color=INK,
                    arrowprops=None if ly == y else dict(arrowstyle='-', color=MUTED, lw=0.5))
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_xlabel(f"PC1 ({100 * m['var_explained'][0]:.0f}% of variance)   →  more Assistant-like", color=MUTED, fontsize=8)
    ax.set_ylabel(f"PC2 ({100 * m['var_explained'][1]:.0f}%)", color=MUTED, fontsize=8)
    ax.set_title(f"Persona space so far: {m['n_roles']} roles, PC1 vs PC2", loc='left', color=INK, fontsize=10.5, pad=10)
    fig.text(0.01, 0.01, f"Layer {m['layer']}; darker = further toward the Assistant on the calibration axis; "
             f"cos(PC1, calibration axis) = {m['cos_pc1_calib_axis']:.2f}.\nPreliminary: roles finished so far "
             f"(needs ≥{m['min_score3']} answers fully in role); the full 276-role run is in progress.",
             color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(ROOT / 'figures/persona_map.png', facecolor=SURFACE); plt.close(fig)


def in_character():
    """How often each finished role is played fully in character (judge score 3)."""
    m = json.load(open(ROOT / 'results/persona_map_partial.json'))
    rate = sorted(((k, v) for k, v in m['frac_score3'].items() if v is not None), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7.2, 0.19 * len(rate) + 1.0), dpi=200, facecolor=SURFACE); style(ax)
    ax.barh(range(len(rate)), [100 * v for _, v in rate], height=0.62, color=BLUE)
    for i, (k, v) in enumerate(rate):
        ax.text(100 * v + 1, i, f'{100 * v:.0f}%', va='center', fontsize=6.5, color=MUTED)
    ax.set_yticks(range(len(rate)), [k for k, _ in rate], color=INK, fontsize=7)
    ax.set_xlim(0, 108); ax.set_xticks([0, 25, 50, 75, 100], ['0', '25', '50', '75', '100%'])
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_title('How often each role is played fully in character', loc='left', color=INK, fontsize=10.5, pad=10)
    fig.text(0.01, 0.005, 'Share of 120 answers per role with judge score 3. Roles finished so far in the full run.',
             color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.015, 1, 1)); fig.savefig(ROOT / 'figures/in_character.png', facecolor=SURFACE); plt.close(fig)


def deception_layers():
    """Within-prompt AUROC (lie vs honest, same prompt) by layer: learned vs instructed direction vs baselines."""
    r = json.load(open(ROOT / 'results/deception_report_qwen3_8b.json'))
    E = r['eval_by_layer']; L = range(len(E))
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200, facecolor=SURFACE); style(ax)
    nat = [e['natural_oof']['within_prompt_auroc'] for e in E]
    ins = [e['instructed']['within_prompt_auroc'] for e in E]
    ln, en = E[0]['length']['within_prompt_auroc'], E[0]['entropy']['within_prompt_auroc']
    ax.fill_between([0, len(E) - 1], 0.75, 1.0, color='#e4f0e8', zorder=0, lw=0)
    ax.text(0.3, 0.76, 'pre-registered pass bar (0.75)', fontsize=7.5, color=MUTED, va='bottom')
    for y, name in [(ln, 'response length'), (en, 'token entropy'), (0.5, 'chance')]:
        ax.hlines(y, 0, len(E) - 1, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.text(len(E) - 0.5, y, ' ' + name, fontsize=7.5, color=MUTED, va='center')
    ax.plot(L, nat, color=CAT[0], lw=2); ax.text(len(E) - 0.5, nat[-1] + 0.012, ' learned from real lies', fontsize=8, color=INK, va='center')
    ax.plot(L, ins, color=CAT[1], lw=2); ax.text(len(E) - 0.5, ins[-1] - 0.012, ' instructed ("lie to the user")', fontsize=8, color=INK, va='center')
    ax.axvline(r['layer'], color=GRID, lw=1)
    ax.text(r['layer'] + 0.3, 0.46, f"layer {r['layer']} (fixed in advance)", fontsize=7.5, color=MUTED)
    ax.set_ylim(0.45, 0.85); ax.set_xlim(0, len(E) + 11)
    ax.set_xticks([0, 6, 12, 18, 24, 30, 35]); ax.set_xlabel('layer', color=MUTED, fontsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_title('Telling real lies from honest answers to the same prompt', loc='left', color=INK, fontsize=10.5, pad=10)
    fig.text(0.01, 0.01, f"Qwen3-8B on MASK: {r['n_lie']} lies vs {r['counts']['labels']['honest']} honest answers, "
             f"{r['n_mixed_prompts']} prompts with both.\nAUROC computed within each prompt (topic held fixed); the learned direction is scored out-of-fold.",
             color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(ROOT / 'figures/deception_by_layer.png', facecolor=SURFACE); plt.close(fig)


if __name__ == '__main__':
    capping(); persona_map(); in_character(); deception_layers()
    print('wrote figures/*.png')
