"""Judge role adherence with blind Claude subagents instead of the GPU judge (Qwen3-14B-AWQ).

  python3 full/judge_local.py todo             # roles with responses but no scores
  python3 full/judge_local.py prep ROLE...     # write full/judge_work/<role>.md (rubric + numbered answers)
  python3 full/judge_local.py ingest ROLE...   # read full/judge_work/<role>.scores.json -> full/state/scores/<role>.json

The packet holds the role's eval_prompt rubric (same text 3_judge.py sends) and the answers, keyed exactly like
3_judge.py ('pos_p{prompt}_q{question}'), so 2_activations.py's score==3 filter works unchanged. The judge
sees no system prompt, no project context and no other role.
"""
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE, WORK = ROOT / 'state', ROOT / 'judge_work'
CODE_ZIP = ROOT.parent / 'dataset' / 'assistant-axis-code.zip'   # upstream pipeline + role data


def rubric(role):
    import zipfile
    with zipfile.ZipFile(CODE_ZIP) as z:
        t = json.loads(z.read(f'data/roles/instructions/{role}.json'))['eval_prompt']
    head = t.split('Prompt:')[0].strip()
    tail = t.split('[ANSWER END]')[-1].strip()
    return head, tail


def key(e):
    return f"{e['label']}_p{e['prompt_index']}_q{e['question_index']}"


def todo():
    r = sorted(Path(f).stem for f in glob.glob(str(STATE / 'responses' / '*.jsonl')))
    sc = lambda x: STATE / 'scores' / f'{x}.json'
    return [x for x in r if not x.startswith('default') and not (sc(x).exists() and sc(x).stat().st_size > 2)]


def prep(role):
    WORK.mkdir(exist_ok=True)
    head, tail = rubric(role)
    es = [json.loads(l) for l in open(STATE / 'responses' / f'{role}.jsonl')]
    out = [f'# Judging task: {role}\n', head, '',
           'For EACH item below, apply this rule to its QUESTION and ANSWER:', '', tail.split('Respond with a number between 0 and 3')[0].strip(), '',
           f'There are {len(es)} items. Score every one independently.', '']
    for e in es:
        out += [f'## ITEM {key(e)}', '[QUESTION START]', e['question'], '[QUESTION END]',
                '[ANSWER START]', e['conversation'][-1]['content'], '[ANSWER END]', '']
    (WORK / f'{role}.md').write_text('\n'.join(out))
    return len(es)


def ingest(role):
    es = [json.loads(l) for l in open(STATE / 'responses' / f'{role}.jsonl')]
    want = {key(e) for e in es}
    got = json.load(open(WORK / f'{role}.scores.json'))
    bad = {k: v for k, v in got.items() if v not in (0, 1, 2, 3)}
    miss = want - set(got)
    assert not bad and not miss and set(got) <= want, f'{role}: bad={bad} missing={sorted(miss)[:5]} extra={sorted(set(got) - want)[:5]}'
    (STATE / 'scores').mkdir(exist_ok=True)
    json.dump({k: got[k] for k in sorted(got)}, open(STATE / 'scores' / f'{role}.json', 'w'))
    return sum(v == 3 for v in got.values()), len(got)


if __name__ == '__main__':
    cmd, roles = sys.argv[1], sys.argv[2:]
    if cmd == 'todo':
        print(' '.join(todo()))
    elif cmd == 'prep':
        for r in roles or todo():
            print(r, prep(r))
    elif cmd == 'ingest':
        for r in roles:
            print(r, 'score3 %d/%d' % ingest(r))
