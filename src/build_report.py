"""Build the English final report exclusively from complete saved experiments.

Run: .venv/bin/python -m src.build_report
The editable narrative is also written to docs/report.md. Change this generator
for reproducible editorial edits; no training is required. The completeness gate
refuses to create a final PDF until each frozen model has exactly 500 full
held-out episodes, a matching hash, and every required plot exists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = {
    'cartpole_basic': 'runs/cartpole_basic',
    'pong_basic': 'runs/pong_basic',
    'cartpole_baseline': 'runs/cartpole_baseline',
    'pong_baseline': 'runs/pong_baseline',
}
NAMES = {
    'cartpole_basic': 'CartPole basic', 'pong_basic': 'Pong basic',
    'cartpole_baseline': 'CartPole baseline', 'pong_baseline': 'Pong baseline',
}
NAVY = colors.HexColor('#18334B')
TEAL = colors.HexColor('#007E87')
GRAY = colors.HexColor('#53616C')
PALE = colors.HexColor('#EDF4F6')


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def truth(value):
    return value is True or str(value).lower() == 'true'


def number(value, decimals=3):
    value = float(value)
    if abs(value) < 1e-3 and value:
        return f'{value:.2e}'
    return f'{value:,.{decimals}f}'


def seconds(value):
    value = float(value)
    return f'{value / 60:.1f} min' if value >= 60 else f'{value:.2f} s'


def load_inputs(run_map):
    errors, runs = [], {}
    for key, folder_name in run_map.items():
        folder = ROOT / folder_name
        required = ['config.json', 'train.csv', 'evaluation.csv',
                    'evaluation_summary.json', 'FROZEN.json', 'final.pt']
        missing = [name for name in required if not (folder / name).exists()]
        if missing:
            errors.append(f'{folder_name}: missing {", ".join(missing)}')
            continue
        cfg = read_json(folder / 'config.json')
        if cfg['game'] != key.split('_')[0] or bool(cfg['baseline']) != key.endswith('baseline'):
            errors.append(f'{folder_name}: config does not match canonical method/environment')
        train, evaluation = read_csv(folder / 'train.csv'), read_csv(folder / 'evaluation.csv')
        frozen = read_json(folder / 'FROZEN.json')
        summary = read_json(folder / 'evaluation_summary.json')
        if len(evaluation) != 500:
            errors.append(f'{folder_name}: {len(evaluation)} evaluation rows, require exactly 500')
        if len(train) != int(cfg['episodes']):
            errors.append(f'{folder_name}: training count differs from formal budget')
        if sorted(int(row['episode']) for row in evaluation) != list(range(500)):
            errors.append(f'{folder_name}: evaluation IDs must be 0..499 once each')
        if any(not (truth(row['terminated']) or truth(row['truncated'])) for row in evaluation):
            errors.append(f'{folder_name}: at least one evaluation row is not a completed environment episode')
        if any(int(row['seed']) != int(cfg['eval_seed_base']) + int(row['episode']) for row in evaluation):
            errors.append(f'{folder_name}: evaluation seed mapping differs from config')
        train_seeds = {int(row['seed']) for row in train}
        if train_seeds & {int(row['seed']) for row in evaluation}:
            errors.append(f'{folder_name}: final seeds overlap training seeds')
        checksum = digest(folder / 'final.pt')
        if checksum != frozen['sha256'] or checksum != summary['checkpoint_sha256']:
            errors.append(f'{folder_name}: frozen checkpoint checksum mismatch')
        if any(row['checkpoint'] != checksum for row in evaluation):
            errors.append(f'{folder_name}: an evaluation row refers to another checkpoint')
        rewards = np.array([float(row['reward']) for row in evaluation])
        if len(rewards) > 1 and (not np.isclose(rewards.mean(), summary['mean']) or
                                not np.isclose(rewards.std(ddof=1), summary['std_sample'])):
            errors.append(f'{folder_name}: saved evaluation statistics differ from raw CSV')
        if int(summary['n']) != 500:
            errors.append(f'{folder_name}: summary n != 500')
        for kind in ('training', 'evaluation'):
            if not (ROOT / f'figures/{key}_{kind}.png').exists():
                errors.append(f'missing figures/{key}_{kind}.png')
        runs[key] = dict(folder=folder_name, cfg=cfg, train=train, evaluation=evaluation,
                         frozen=frozen, summary=summary, sha256=checksum)
    for path in ['data/shuttle/summary.json', 'data/shuttle/validation.json',
                 'data/validation/rl.json', 'data/environment.json', 'data/benchmark/decision.json',
                 'figures/shuttle_enumeration.png', 'figures/shuttle_value_iteration.png',
                 'figures/shuttle_policy_iteration.png']:
        if not (ROOT / path).exists():
            errors.append(f'missing {path}')
    if errors:
        raise ValueError('Final report completeness gate failed:\n- ' + '\n- '.join(errors))
    return runs


class Report:
    """One content stream for PDF and an editable Markdown equivalent."""
    def __init__(self):
        styles = getSampleStyleSheet()
        self.styles = {
            'title': ParagraphStyle('Title2', parent=styles['Title'], fontName='Helvetica-Bold',
                fontSize=25, leading=29, textColor=NAVY, spaceAfter=16),
            'h1': ParagraphStyle('Heading12', parent=styles['Heading1'], fontName='Helvetica-Bold',
                fontSize=17, leading=20, textColor=NAVY, spaceAfter=12),
            'h2': ParagraphStyle('Heading22', parent=styles['Heading2'], fontName='Helvetica-Bold',
                fontSize=11.2, leading=14, textColor=TEAL, spaceBefore=10, spaceAfter=5),
            'body': ParagraphStyle('Body2', parent=styles['BodyText'], fontName='Helvetica',
                fontSize=9.4, leading=13.3, textColor=NAVY, spaceAfter=7),
            'caption': ParagraphStyle('Caption2', parent=styles['BodyText'], fontName='Helvetica',
                fontSize=8.2, leading=11, textColor=GRAY, spaceAfter=10),
            'small': ParagraphStyle('Small2', parent=styles['BodyText'], fontName='Helvetica',
                fontSize=8.0, leading=10.3, textColor=NAVY, spaceAfter=5),
            'code': ParagraphStyle('Code2', parent=styles['Code'], fontName='Courier',
                fontSize=7.25, leading=10.2, textColor=NAVY, backColor=PALE, borderPadding=6,
                spaceBefore=4, spaceAfter=9, splitLongWords=True),
            'formula': ParagraphStyle('Formula2', parent=styles['BodyText'], fontName='Courier',
                fontSize=8.1, leading=11.8, textColor=NAVY, backColor=PALE, borderPadding=8,
                spaceBefore=5, spaceAfter=10),
        }
        self.story, self.md = [], []

    def heading(self, text, level=1):
        self.story.append(Paragraph(escape(text), self.styles['h1' if level == 1 else 'h2']))
        self.md += ['#' * (level + 1) + ' ' + text, '']

    def paragraph(self, text, kind='body'):
        self.story.append(Paragraph(escape(text), self.styles[kind]))
        self.md += [text, '']

    def formula(self, text):
        self.story.append(Paragraph(escape(text).replace('\n', '<br/>'), self.styles['formula']))
        self.md += ['```text', text, '```', '']

    def code(self, lines):
        text = '\n'.join(lines) if isinstance(lines, list) else lines
        self.story.append(Paragraph(escape(text).replace('\n', '<br/>'), self.styles['code']))
        self.md += ['```sh', text, '```', '']

    def table(self, headers, rows, widths=None):
        data = [[Paragraph(escape(str(cell)), self.styles['small']) for cell in headers]]
        data += [[Paragraph(escape(str(cell)), self.styles['small']) for cell in row] for row in rows]
        table = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PALE),
            ('TEXTCOLOR', (0, 0), (-1, 0), NAVY),
            ('LINEBELOW', (0, 0), (-1, 0), .6, TEAL),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LINEBELOW', (0, 1), (-1, -1), .2, colors.HexColor('#D7E0E4')),
        ]))
        self.story += [table, Spacer(1, 8)]
        self.md += ['| ' + ' | '.join(map(str, headers)) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
        self.md += ['| ' + ' | '.join(str(c).replace('|', '/') for c in row) + ' |' for row in rows] + ['']

    def figure(self, name, caption, max_height=2.7 * inch):
        path = ROOT / 'figures' / name
        with PILImage.open(path) as im:
            w, h = im.size
        width = min(7.12 * inch, max_height * w / h)
        height = width * h / w
        image = Image(str(path), width=width, height=height, hAlign='CENTER')
        self.story += [image, Paragraph(escape(caption), self.styles['caption'])]
        self.md += [f'![{caption}](../figures/{name})', '']

    def page(self):
        self.story.append(PageBreak())
        self.md += ['<!-- page break -->', '']

    def link(self, title, url):
        self.story.append(Paragraph(f'<link href="{escape(url)}" color="#007E87">{escape(title)}</link>', self.styles['small']))
        self.md += [f'[{title}]({url})', '']


def repository_info():
    """Resolve actual origin, or retained verified metadata for an unzipped copy."""
    saved_path=ROOT/'data/repository.json'
    saved=read_json(saved_path) if saved_path.exists() else None
    try:
        proc=subprocess.run(['git','-C',str(ROOT),'remote','get-url','origin'],capture_output=True,text=True)
        origin=proc.stdout.strip() if proc.returncode==0 else ''
    except FileNotFoundError:
        origin=''
    if origin:
        match=re.search(r'github\.com[:/]([A-Za-z0-9_-]+)/([A-Za-z0-9_.-]+)$',origin.rstrip('/'))
        if not match:
            raise ValueError('Present origin is not a recognized GitHub repository URL')
        owner,repo=match.groups()
        repo=repo.removesuffix('.git')
        url=f'https://github.com/{owner}/{repo}'
        if saved:
            saved_owner=saved.get('owner') or saved.get('authenticated_login')
            if str(saved_owner).lower()!=owner.lower() or saved.get('repo')!=repo or saved.get('url','').rstrip('/')!=url:
                raise ValueError('Git origin disagrees with retained verified repository metadata')
        return owner,repo,url
    if not saved:
        raise ValueError('No origin or retained verified data/repository.json available')
    owner=saved.get('owner') or saved.get('authenticated_login')
    repo=saved.get('repo');url=saved.get('url','').rstrip('/')
    if not owner or not repo or url!=f'https://github.com/{owner}/{repo}' or not saved.get('authenticated_login') or not saved.get('verification'):
        raise ValueError('Retained GitHub metadata is missing its verified identity or consistent URL')
    return owner,repo,url


def build_content(runs):
    r = Report()
    shuttle = read_json(ROOT / 'data/shuttle/summary.json')
    sv = read_json(ROOT / 'data/shuttle/validation.json')
    rv = read_json(ROOT / 'data/validation/rl.json')
    env = read_json(ROOT / 'data/environment.json')
    bench = read_json(ROOT / 'data/benchmark/decision.json')
    owner, repo, url = repository_info()
    tests = {x['name']: x for x in sv['tests']}
    total_evaluation_steps = sum(int(x['summary']['env_steps']) for x in runs.values())
    total_training_steps = sum(int(x['train'][-1]['env_steps_total']) for x in runs.values())

    # Page 1
    r.story.append(Spacer(1, .18 * inch))
    r.story.append(Paragraph('Reinforcement Learning<br/>Assignment 1', r.styles['title']))
    r.md += ['# Reinforcement Learning Assignment 1', '']
    r.paragraph('Yujun Sun | October 2026')
    r.paragraph('This report studies shuttle dispatching using finite-horizon enumeration, value iteration, and policy iteration. It also compares Monte Carlo REINFORCE with and without a scalar baseline on CartPole-v1 and Pong-v5, examining learning performance, gradient variance, and algorithm correctness.')
    r.heading('Code and data', 2)
    r.paragraph(f'GitHub username: {owner}. Repository: {repo}.')
    r.link('GitHub repository: ' + url, url)
    r.heading('Evaluation summary', 2)
    result_rows = []
    for key, x in runs.items():
        s = x['summary']
        result_rows.append([NAMES[key], len(x['train']), s['n'], number(s['mean']), number(s['std_sample'])])
    r.table(['Model', 'Train episodes', 'Eval episodes', 'Mean reward', 'Sample SD'], result_rows,
            [145, 75, 75, 100, 100])
    r.paragraph(f'The experiments comprise {total_training_steps:,} training steps and 2,000 held-out evaluation episodes totaling {total_evaluation_steps:,} steps. Evaluation samples actions from each fixed final policy. Checkpoints are selected at the predetermined training endpoint, independently of evaluation rewards.')
    r.paragraph('Reward means and sample standard deviations describe raw undiscounted complete-episode totals. They are distinct from the discounted optimization objective and from the variance of a gradient estimator. All four models are single-training-seed experiments; comparisons describe these runs rather than population-level causal effects.')
    r.page()

    # Page 2
    r.heading('1. Shuttle model and finite enumeration')
    r.paragraph('State s is the observed queue size in {0,...,200} at a decision epoch. Action a in {0,1} means wait or dispatch one shuttle. Define q(s,a)=max(s-15a,0). The timing is: observe s, choose a, serve up to 15 passengers, charge dispatch plus holding for passengers still waiting, then add an independent uniform arrival A in {1,...,5}. New arrivals are first eligible for service at the next epoch.')
    r.formula('c(s,a) = 100 a + 2 max(s - 15 a, 0)\ns_next = min(200, max(s - 15 a, 0) + A)\nP(s_next=j | s,a) = (1/5) sum_{A=1}^5 1[j=s_next(s,a,A)]\nV_pi(s) = E_pi[sum_{t=0}^infinity 0.95^t c(s_t,a_t) | s_0=s]')
    r.paragraph('Value functions represent expected discounted costs, which are minimized. Empty or underfilled dispatch is allowed, costs 100, and serves min(s,15) passengers. At most one shuttle leaves per epoch. Arrivals above the queue cap of 200 are discarded without an overflow penalty; equal clipped successor states accumulate probability. Underfilled service and overflow handling are modeling assumptions.')
    r.paragraph('Finite-horizon enumeration considers both actions and all five arrival outcomes through backward recursion, with T=500 and terminal value V_T(s)=0. Reusing state values at each time step avoids exponential enumeration of complete trajectories.')
    r.formula('V_t(s) = min_a { c(s,a) + 0.95 sum_j P(j|s,a) V_{t+1}(j) }\nV_500(s)=0; t=499,...,0')
    r.figure('shuttle_enumeration.png', 'Figure 1. Finite-horizon enumeration: V_0(s) for T=500, terminal continuation cost zero.', 2.7 * inch)
    r.paragraph(f'At s=0 the computed cost is {number(shuttle["values_at_states"]["0"]["finite"],6)}; at s=200 it is {number(shuttle["values_at_states"]["200"]["finite"],6)}. More initial waiting increases cost through holding and the number of successive dispatches required.', 'small')
    r.page()

    # Page 3
    r.heading('2. Infinite-horizon shuttle solutions')
    r.paragraph('Value iteration starts at V=0 and repeatedly applies the minimum-cost Bellman operator until the maximum successive change is at most 1e-10. The contraction factor is 0.95. Policy iteration starts with never dispatching, solves (I-gamma P_pi)V=c_pi in float64, and improves all actions until the policy stops changing. Ties are resolved toward wait.')
    r.figure('shuttle_value_iteration.png', f'Figure 2. Value iteration converged in {shuttle["value_iteration"]["iterations"]} iterations; the values are discounted costs.', 2.55 * inch)
    threshold = shuttle['optimal_policy']['threshold']
    r.figure('shuttle_policy_iteration.png', f'Figure 3. Policy iteration converged in {shuttle["policy_iteration"]["iterations"]} improvements/evaluations: wait below {threshold}, dispatch from {threshold} through 200.', 2.55 * inch)
    r.paragraph(f'The resulting threshold {threshold} is below capacity 15. Waiting for a full shuttle is not generally optimal because accumulated holding cost can outweigh unused capacity. The numerical minimum action-value gap is {number(sv["minimum_optimal_action_gap"],6)}, attained at state {sv["minimum_gap_state"]}; the selected policy is not a near-tie artifact.', 'small')
    r.page()

    # Page 4
    r.heading('3. Shuttle validation and edge cases')
    r.code('.venv/bin/python src/shuttle.py all\n.venv/bin/python src/shuttle.py validate')
    descriptions = {
        'independent_small_tree': ('Independent small decision tree', 'A separate recursive exhaustive tree must match the matrix recurrence for all states and horizons.', f'{tests["independent_small_tree"]["result"]["comparisons"]} comparisons; max error {number(tests["independent_small_tree"]["result"]["max_absolute_error"])}.'),
        'hand_calculated_boundaries': ('Hand-derived timing and boundaries', 'Empty/underfilled service, K, K+1, and cap aggregation must match manual successors/costs.', f'Max error {number(tests["hand_calculated_boundaries"]["result"]["max_absolute_error"])} in 8 cases.'),
        'deterministic_zero_arrivals': ('Deterministic no-arrival control', 'With A=0, enumerate the number of immediate dispatches followed by waiting forever.', f'Max value error {number(tests["deterministic_zero_arrivals"]["result"]["max_absolute_error"])}.'),
        'optimality_residual_and_solver_agreement': ('Optimality certificate and solver agreement', 'Small Bellman residual implies small fixed-point error; VI and PI must agree.', f'VI residual {number(shuttle["value_iteration"]["bellman_residual"])}; PI {number(shuttle["policy_iteration"]["bellman_residual"])}.'),
        'finite_infinite_tail_bound': ('Finite/infinite horizon tail', 'Nonnegative omitted tail bounds V_infinity - V_0 at T=500.', f'Max difference {number(tests["finite_infinite_tail_bound"]["result"]["max_finite_infinite_difference"])}.'),
        'never_dispatch_reference': ('Nonoptimal policy reference', 'Optimal costs cannot exceed a fixed never-dispatch policy.', f'At s=200 cost improves by {number(tests["never_dispatch_reference"]["result"]["improvement_at_200"])}.'),
    }
    r.table(['Test purpose', 'Theory and running method', 'Observed result'], list(descriptions.values()), [125,235,135])
    r.paragraph('The small-tree oracle uses eight states and horizons 0 through 4, recursively visiting every action/outcome subtree without transition tensors or memoization. It therefore tests the expectation/minimization independently of the main solver. The no-arrival test changes the original arrival law only for validation and compares against an analytic discounted dispatch/wait cost.')
    agreement = tests['optimality_residual_and_solver_agreement']['result']
    tail = tests['finite_infinite_tail_bound']['result']
    r.paragraph(f'The maximum VI/PI value difference is {number(agreement["max_vi_pi_value_difference"])} with {agreement["policy_disagreements"]} policy disagreements. The VI residual gives a sup-norm value-error bound residual/(1-gamma)={number(agreement["vi_residual_error_bound"])}. The finite/infinite difference is below the independent geometric tail bound {number(tail["geometric_tail_bound"])}.')
    r.table(['State/action', 'Charged cost', 'Successors or cap probability'], [
        ['0 / wait', '0', '1,2,3,4,5; each 0.2'],
        ['0 / dispatch', '100', 'same successors as waiting; strictly dominated'],
        ['14 or 15 / dispatch', '100', '1,2,3,4,5; each 0.2'],
        ['16 / dispatch', '102', '2,3,4,5,6; each 0.2'],
        ['198 / wait', '396', 'P(199)=0.2; P(200)=0.8'],
        ['200 / dispatch', '470', '186,187,188,189,190; each 0.2'],
    ], [125,100,270])
    r.paragraph('These checks support the stated capped-queue MDP. They do not validate the realism of dropping overflow passengers. A different overflow rule or additional dispatch constraints defines a different decision problem.', 'small')
    r.page()

    # Page 5
    r.heading('4. REINFORCE objective and implementation')
    r.paragraph('For an episode of length L, the training objective discounts rewards from the episode start. G_t is return-to-go with time measured relative to t. Evaluation instead uses R=sum_t r_t without discount. CartPole gamma=0.95; Pong gamma=0.99.')
    r.formula('J(theta) = E_pi[sum_{t=0}^{L-1} gamma^t r_t]\nG_t = sum_{u=t}^{L-1} gamma^(u-t) r_u\nloss = -(1/(100 B)) sum_{episode i=1}^B sum_{t=0}^{L_i-1}\n       gamma^t log pi_theta(a_it | x_it) detach(G_it - b)')
    r.paragraph('The absolute-time gamma^t factor corresponds to the episode-start objective and is separate from the discount inside G_t. Time steps are summed and complete episodes are averaged, with the same fixed scale 100 in every model. The basic estimator uses b=0 and uncentered returns. Adam minimizes the loss, with an aggregate gradient-norm limit of 10.')
    r.table(['Environment', 'Policy input and network', 'Episode completion'], [
        ['CartPole-v1', 'Raw 4-vector -> Linear(4,64) -> Tanh -> Linear(64,2); sampled left/right.', 'Natural failure or official 500-step TimeLimit.'],
        ['ALE/Pong-v5', 'RGB image -> crop/downsample/background mask -> 80x80 frame difference -> Linear(6400,64) -> ReLU -> Linear(64,2). Outputs map to environment actions [2,3].', 'Natural game end or official ALE 108,000-frame cap; no point/rally boundaries.'],
    ], [95,275,125])
    r.paragraph('Networks use PyTorch default nn.Linear random initialization, with matched seeds 101 for CartPole and 202 for Pong. Adam uses betas=(0.9,0.999), eps=1e-8 and weight_decay=0.', 'small')
    r.paragraph(f'The Pong environment is ALE/Pong-v5 in Gymnasium {env["gymnasium"]} with ale-py {env["ale_py"]}. Its six action meanings are {", ".join(env["pong_action_meanings"])}. The policy selects RIGHT or LEFT, mapped to indices 2 and 3, using RGB game observations.')
    r.paragraph('Pong uses frameskip=4, sticky action probability=0.25, mode=0, difficulty=0, and the minimal six-action space. Crop rows 35:195, downsample both dimensions by 2, use the red channel, erase colors 144/109 and map other nonzero pixels to 1. Input is current minus previous binary frame, with a zero difference at reset. No frame stack, FIRE/NOOP wrapper, point-based termination, reward clipping, or custom episode shortening is used.')
    r.paragraph('Rollouts run without autograd; the policy parameters remain fixed for the entire update batch. Observations/actions are stored on CPU (Pong observations as int8), then log probabilities are recomputed in bounded chunks of 512. Gradients accumulate over all chunks; exactly one optimizer step occurs after the entire batch. Thus recomputation uses the sampling policy and does not retain long episode graphs. Evaluation uses inference mode with no gradient recording.')
    r.page()

    # Page 6
    r.heading('5. Experimental setup and training budgets')
    r.paragraph(f'Experiments ran on an Apple M4 Pro MacBook Pro with 24 GB unified memory, macOS 15.6.1, native arm64 Python 3.12.14, and PyTorch {env["torch"]}. Shuttle dispatching and CartPole use the CPU. Pong device selection compares CPU and MPS performance.')
    r.paragraph('Pong CPU/MPS timing uses the same 64-unit policy, image preprocessing, complete-episode environment configuration, seed sequence, batch size, loss, and Adam update. Throughput includes rollout, Python environment stepping, preprocessing, inference, return calculation, and updates. Device selection uses measured end-to-end environment steps per second rather than matrix throughput alone.')
    br = []
    for x in bench['summaries']:
        br.append([x['device'], seconds(x['seconds']), f'{x["environment_steps"]:,}', number(x['steps_per_second'],1), seconds(x['update_seconds_mean']), f'{x["process_peak_rss_bytes"] / 2**20:.1f} MiB'])
    r.table(['Device', 'Duration', 'Env steps', 'Steps/s', 'Mean update', 'Peak RSS'], br, [55,75,85,75,95,110])
    r.paragraph(f'Pong uses {bench["selected"].upper()}, which achieved the higher end-to-end throughput. Each device was measured in a separate process; peak memory is the process RSS high-water mark.', 'small')
    rr=[]
    for key,x in runs.items():
        cfg=x['cfg']; t=x['train'][-1]
        rr.append([NAMES[key],cfg['seed'],cfg['episodes'],cfg['batch_size'],cfg['lr'],cfg['device'],seconds(t['elapsed_seconds'])])
    r.table(['Run', 'Seed', 'Episodes', 'Batch', 'Adam LR', 'Device', 'Train time'], rr, [125,45,65,45,65,55,95])
    pilot=[]
    for folder in sorted((ROOT/'runs').glob('pilot*')):
        if (folder/'config.json').exists() and (folder/'train.csv').exists():
            cfg=read_json(folder/'config.json'); rows=read_csv(folder/'train.csv')
            if rows: pilot.append([folder.name,len(rows),cfg['lr'],number(np.mean([float(z['reward']) for z in rows[-100:]])),seconds(rows[-1]['elapsed_seconds'])])
    if pilot:
        r.heading('Learning-rate pilots',2)
        r.table(['Pilot', 'Episodes', 'Adam LR', 'Last up-to-100 mean', 'Time'],pilot,[160,65,65,115,90])
    high_path=ROOT/'runs/pilot_pong_lr0p01/pilot_diagnostic.json'
    low_path=ROOT/'runs/pilot_pong_lr0p001/pilot_diagnostic.json'
    if high_path.exists() and low_path.exists():
        high=read_json(high_path); low=read_json(low_path)
        r.paragraph(f'Pong uses Adam 0.001 after 128-episode matched tuning pilots: mean diagnostic action entropy was {number(high["mean_action_entropy_nats"],4)} nats at 0.01 and {number(low["mean_action_entropy_nats"],4)} at 0.001. Both lost; neither saturated above probability .999. The lower rate conservatively preserves exploration. Eight diagnostic games per rate are weak single-seed evidence, not proof of final superiority. CartPole retains the default 0.01.', 'small')
    clipped_counts=[]
    for key,x in runs.items():
        updates=read_csv(ROOT/x['folder']/'updates.csv')
        count=sum(float(z['gradient_norm']) > float(x['cfg']['grad_clip']) for z in updates)
        clipped_counts.append(f'{NAMES[key]} {count}/{len(updates)}')
    r.paragraph('Gradient clipping bound 10 was monitored: clipped updates / all updates = ' + '; '.join(clipped_counts) + '. A zero count means clipping did not alter the updates in that run.', 'small')
    r.paragraph('Each model trains for 4,096 complete episodes, comprising 512 updates of eight episodes. Pilot and benchmark episodes are excluded from this budget. The policy after the final update is evaluated on 500 complete episodes using seeds independent of training and tuning. This fixed stopping rule does not use evaluation rewards for checkpoint selection.')
    r.paragraph('Batch-boundary checkpoints store the policy, Adam state, baseline, episode and step counts, and seed mappings for reproducible resumption.', 'small')
    r.page()

    # Pages 7-10: exactly eight required PG/baseline figure contents.
    figure_number=4
    for key in ['cartpole_basic','pong_basic','cartpole_baseline','pong_baseline']:
        x=runs[key]; cfg=x['cfg']; s=x['summary']; tr=x['train']
        title=NAMES[key]
        r.heading(f'{6 + list(["cartpole_basic","pong_basic","cartpole_baseline","pong_baseline"]).index(key)}. {title}: results')
        train_rewards=np.array([float(z['reward']) for z in tr])
        first=float(train_rewards[:min(100,len(tr))].mean()); last=float(train_rewards[-min(100,len(tr)):].mean())
        discounted=np.array([float(z['discounted_return']) for z in tr])
        first_j=float(discounted[:min(100,len(tr))].mean());last_j=float(discounted[-min(100,len(tr)):].mean())
        r.paragraph(f'{len(tr):,} train episodes; {int(tr[-1]["env_steps_total"]):,} steps; seed {cfg["seed"]}; Adam {cfg["lr"]}; {cfg["device"]}. First to last 100 means: raw R {number(first)} to {number(last)}; discounted J {number(first_j,4)} to {number(last_j,4)}. The moving window grows from 1 to 100, then stays 100.')
        r.figure(f'{key}_training.png',f'Figure {figure_number}. {title}: raw training reward and trailing 100-episode mean.',3.5*inch)
        figure_number+=1
        r.figure(f'{key}_evaluation.png',f'Figure {figure_number}. {title}: 500 frozen-policy complete-episode rewards.',3.5*inch)
        figure_number+=1
        both=int(s.get('both_terminated_and_truncated',sum(truth(z['terminated']) and truth(z['truncated']) for z in x['evaluation'])))
        term_only=int(s.get('terminated_only',int(s['terminated'])-both))
        trunc_only=int(s.get('truncated_only',int(s['truncated'])-both))
        ending_note=' Flags may coincide on CartPole step 500.' if cfg['game']=='cartpole' else ''
        r.paragraph(f'Final mean={number(s["mean"])}; sample SD={number(s["std_sample"])} (ddof=1); range [{number(s["min"],0)}, {number(s["max"],0)}]; mean length={number(s["mean_length"],1)} steps. Exclusive end cases: natural only {term_only}, time-limit only {trunc_only}, both flags {both}; total 500.'+ending_note, 'small')
        r.page()

    # Page 11
    r.heading('10. Policy-gradient validation')
    r.code('.venv/bin/python -m src.validate_rl --output data/validation/rl.json')
    ret=rv['returns']; exact=rv['exact_gradient_and_baseline']; auto=rv['autograd']; trace=rv['cartpole_trace']
    r.heading('Hand trajectory and objective weighting',2)
    r.paragraph(f'Purpose: detect return indexing, terminal errors and objective mismatch. Theory: rewards [1,2,3], gamma=.95 yield G=[5.6075,4.85,3], with objective weights [1,.95,.9025]. Method: compare the trainer recurrence against direct sums and hand values, including gamma=0/1, empty and one-step cases. Actual maximum error is {number(ret["absolute_max_error"])}. Interpretation: terminal continuation is zero and the final reward is included once.')
    r.heading('Exact trajectory enumeration and update direction',2)
    r.paragraph(f'Purpose: validate the stochastic estimator without Monte Carlo error. Theory: three independent Bernoulli actions with p=.35 and rewards [1+2a0,-.5+3a1,2-a2] have dJ/dtheta=p(1-p)(2+3 gamma-gamma^2)={number(exact["exact_gradient"],8)}. Method: enumerate all eight trajectories and exact probabilities. Actual mean estimator is {number(exact["enumerated_expected_gradient"],8)}. Omitting gamma^t instead gives {number(exact["missing_gamma_t_expected_gradient"],8)}, a bias of {number(exact["missing_gamma_t_bias"],8)}.')
    r.paragraph(f'The float64 PyTorch policy-loss derivative is {number(auto["loss_gradient"],8)}; the central finite-difference objective derivative is {number(auto["finite_difference_objective_gradient"],8)}. One small SGD step raises J from {number(auto["objective_before"],6)} to {number(auto["objective_after"],6)}. This verifies the loss sign, not Adam convergence in Pong.')
    r.heading('CartPole first-ten-step trace',2)
    r.paragraph('Purpose: connect the return equations to real state/action/reward transitions. A separate uniform policy uses CartPole reset seed 74219 and Python action seed 74220. It runs the entire episode before returns are calculated; displaying ten rows does not truncate training or evaluation.')
    trace_rows=trace.get('first_ten_steps',trace.get('trajectory',trace.get('full_trajectory',[])))[:10]
    if not trace_rows:
        raise ValueError('Validation JSON lacks the real ten-step trajectory')
    r.table(['t','Observation (x, xdot, angle, angledot)','a','r','Return-to-go'],[
        [z['step'],', '.join(f'{float(v):.4f}' for v in z['observation']),z['action'],z['reward'],number(z['return_to_go'],6)] for z in trace_rows
    ],[25,305,25,30,110])
    trace_len=trace.get('episode_length',trace.get('length'))
    trace_reward=trace.get('episode_reward',trace.get('total_reward'))
    r.paragraph(f'The complete diagnostic episode length is {trace_len}; total reward is {trace_reward}. Its full states, actions, rewards, termination flags, discounted weights and returns are saved in data/validation/rl.json. For +1 CartPole rewards, the geometric return formula independently checks the recurrence. This diagnostic is excluded from the formal evaluation.', 'small')
    r.page()

    # Production path + empirical diagnostics: checks the actual trainer, not a mirrored toy only.
    r.heading('11. Implementation validation')
    prod = rv['production_trainer']
    r.paragraph('The implementation tests exercise the policy-gradient estimator, chunked updates, baseline timing, preprocessing, and checkpoint recovery through the training function src.rl.update.')
    r.paragraph(prod['theoretical_expectation'])
    r.paragraph('Method: ' + prod['method'])
    r.table(['Validation check', 'Quantitative result', 'Interpretation'], [
        ['Return recurrence', number(prod['production_returns_max_absolute_error']) + ' max absolute error', 'The production function agrees with direct hand sums.'],
        ['Exact expected update, b=0', number(prod['enumerated_production_update'][0]['actual_theta_after_update'],10) + ' vs ' + number(prod['enumerated_production_update'][0]['expected_theta_after_update'],10), 'Loss sign, gamma weights, fixed scale and batch average agree.'],
        ['Exact expected update, b=2', number(prod['enumerated_production_update'][1]['actual_theta_after_update'],10) + ' vs ' + number(prod['enumerated_production_update'][1]['expected_theta_after_update'],10), 'A fixed baseline preserves the expected update.'],
        ['EMA next batch', number(prod['enumerated_production_update'][1]['baseline_next'],10) + ' vs theory ' + number(prod['enumerated_production_update'][1]['expected_baseline_next'],10), 'The update uses b=2, then assigns the completed-batch target to the next batch.'],
        ['Chunked vs independent graph', number(prod['chunk1_vs_independent_single_graph_parameter_max_absolute_error']) + ' parameter max error; ' + number(prod['chunk1_vs_independent_single_graph_loss_absolute_error']) + ' loss error', 'Even chunk size 1 reproduces the independent full graph within float32 precision.'],
        ['Checkpoint roundtrip', 'Next rollouts identical=' + str(prod['resume_rollouts_identical']) + '; parameter max error=' + number(prod['resume_parameter_max_absolute_error']), 'Saved model, Adam and baseline reproduce an actual next CPU update.'],
        ['Image semantics', 'Test mask pixels=' + str(prod['preprocessing_test_pixels']) + '; signed difference range=' + str(prod['signed_image_difference_range']), 'Background removal and negative motion pixels survive preprocessing without mutating the raw image.'],
    ],[110,190,195])
    r.paragraph('CartPole rollout lengths in the checkpoint experiment were ' + ', '.join(str(x) for x in prod['actual_rollout_episode_lengths']) + '. Resumed and uninterrupted runs produced identical trajectories, parameter updates, and baseline values. The comparison covers CPU execution at complete update boundaries.')
    diagnostic_path = ROOT / 'data/validation/gradient_diagnostic.json'
    if diagnostic_path.exists():
        diagnostic = read_json(diagnostic_path)
        r.page()
        r.heading('12. Fixed-policy gradient variance')
        if diagnostic.get('status') != 'complete' or len(diagnostic.get('models', [])) != 2:
            raise ValueError('Empirical diagnostic JSON is partial; final report must wait for both games')
        r.paragraph('Purpose: measure the implemented saved scalar baseline on actual learned-policy trajectories, and compare task behavior with an uninformed policy. Theory: a scalar fixed before actions preserves the expected gradient but can increase or reduce estimator variance. No optimizer or parameter update occurs; these diagnostics are collected after checkpoints are frozen.')
        r.code('.venv/bin/python -m src.gradient_diagnostic --run runs/cartpole_baseline --run runs/pong_baseline --episodes 32 --device cpu')
        r.formula('g_i(b) = sum_t gamma^t grad log pi(a_it | x_it) (G_it-b)/100\ntrace(sample Cov) = sum_i ||g_i - mean(g)||^2 / (n-1)')
        r.paragraph('Method: use the same complete trajectories for b=0 and the saved EMA, with diagnostic seed base 92000000. Recompute float32 production-policy log probabilities in bounded chunks, before clipping and Adam. Float64 online vector moments avoid storing an episode-by-parameter matrix. Raw paired norms, projections and episode records are retained.')
        gradient_rows=[]
        for model in diagnostic['models']:
            ratio=model['variance_ratio_saved_baseline_over_no_baseline']
            gradient_rows.append([model['game'],model['n'],number(model['saved_baseline'],6),
                number(model['trace_sample_gradient_covariance_no_baseline'],6),
                number(model['trace_sample_gradient_covariance_saved_baseline'],6),
                'undefined' if ratio is None else number(ratio,4)])
        r.table(['Game','n','Saved b','Trace Cov b=0','Trace Cov EMA','Ratio EMA/0'],gradient_rows,[65,30,90,105,105,100])
        for model in diagnostic['models']:
            ratio=model['variance_ratio_saved_baseline_over_no_baseline']
            effect='undefined because the uncentered sample covariance is zero' if ratio is None else ('lower' if ratio<1 else 'higher' if ratio>1 else 'equal')
            r.paragraph(f'{model["game"].title()}: the saved baseline gives {effect} sampled gradient variance in these {model["n"]} episodes. Diagnostic mean raw reward is {number(model["diagnostic_episode_reward_mean"])} and mean length {number(model["diagnostic_episode_length_mean"],1)}. Parameters unchanged={model["parameters_unchanged"]}; natural terminations={model["natural_terminations"]}; official truncations={model["built_in_truncations"]}. A finite-sample difference between mean gradients does not establish bias because empirical baseline scores need not average to zero.', 'small')
        r.heading('Separate uniform random-policy reference',2)
        references=diagnostic.get('random_policy_references',[])
        r.table(['Game','n','Mean raw reward','Sample SD','Mean length'],[[z['game'],z['n'],number(z['reward_mean']),number(z['reward_std_sample']),number(z['mean_episode_length'],1)] for z in references],[80,45,125,120,125])
        r.paragraph('The reference uses zero logits, the same environment/action set and full termination, graph-free collection, and distinct seed base 93000000. Its raw rewards are excluded from all four final evaluation histograms. These small controls support descriptive interpretation; they are not a statistical test of superiority over independent training runs.', 'small')
        cp_models=[z for z in diagnostic['models'] if z['game']=='cartpole']
        if cp_models:
            cp=cp_models[0]
            r.paragraph(f'CartPole explanation: many long-survival trajectories have early return-to-go almost equal to 1/(1-.95)=20, close to the saved b={number(cp["saved_baseline"],7)}. Subtraction removes a nearly constant score-function nuisance. The true episode-start discounted survival gradient can itself be very small because late rewards receive .95^t weight. This explains a very large measured variance reduction without proving a universal baseline advantage. Exact toy identities support unbiasedness; 32 sampled means alone cannot establish or refute it.', 'small')
        r.paragraph('These diagnostics do not use the final held-out 500-episode samples to select a checkpoint. Small diagnostic sample counts support mechanism checks, not definitive ranking of the four trained policies.', 'small')
    r.page()

    # Baseline discussion follows production-path verification.
    r.heading(f'{13 if diagnostic_path.exists() else 12}. Baseline mechanism and variance')
    r.paragraph('The baseline is a scalar EMA carried from earlier complete batches. It is initialized at zero, frozen throughout rollout and the corresponding policy update, and changes only after that update. Its target is the gamma^t-weighted mean return-to-go in the completed batch. It is constant across current states/actions; no value network or critic loss is used.')
    r.formula('target_k = [sum_{i,t} gamma^t G_it] / [sum_{i,t} gamma^t]\nb_(k+1) = 0.9 b_k + 0.1 target_k\nadvantage_it = detach(G_it - b_k)')
    r.paragraph('Because b_k is fixed before any current-batch actions are sampled, its score-function contribution has expectation zero conditional on earlier training history. A simple return-mean target is not the exact variance-minimizing baseline, which would additionally depend on score magnitudes. Thus it can reduce or increase variance, and cannot guarantee a better final score.')
    r.heading('Exact mechanism checks',2)
    r.table(['Fixed b', 'Expected gradient', 'Gradient-estimator variance'],[[z['constant_baseline'],number(z['expected_gradient'],8),number(z['gradient_estimator_variance'],8)] for z in exact['constant_baselines']],[95,200,200])
    same=rv['same_batch_mean_bias']
    r.paragraph(f'Purpose: check action independence and gradient-estimator variance. Theory: fixed b leaves the expected gradient unchanged. Method: exact eight-trajectory enumeration. A baseline b=2 reduces variance by {100*exact["variance_reduction_fraction_at_b2"]:.2f}%; b=8 increases it. An action-dependent baseline instead has expected gradient {number(exact["action_dependent_baseline_expected_gradient"])}, revealing its bias. The detached baseline receives no gradient in automatic differentiation.')
    r.paragraph(f'A separate exact enumeration tests why the current batch is not used for its own mean-centering. With N=4 one-step samples, theory predicts factor 1-1/N=.75. Actual expectation changes from {number(same["unbiased_expected_gradient"],5)} to {number(same["current_batch_mean_expected_gradient"],5)}, measured ratio {number(same["actual_ratio"],5)}. This isolates finite-batch self-dependence; the implemented prior-batch EMA avoids it.')
    r.page()
    r.heading(f'{14 if diagnostic_path.exists() else 13}. Performance comparison and limitations')
    for game in ['cartpole','pong']:
        a=runs[game+'_basic']['summary']; b=runs[game+'_baseline']['summary']
        extra=''
        if game=='pong' and diagnostic_path.exists():
            pd=next(z for z in diagnostic['models'] if z['game']=='pong')
            ratio=pd['variance_ratio_saved_baseline_over_no_baseline']
            if ratio is not None:
                extra=f' The separate paired gradient diagnostic measures {100*(1-ratio):.2f}% lower covariance trace, while raw reward SD is higher. Gradient variance and episode-score variability are different quantities.'
        label='CartPole' if game=='cartpole' else 'Pong'
        r.paragraph(f'{label}: baseline minus basic final mean reward = {number(b["mean"]-a["mean"])}; sample reward-SD change = {number(b["std_sample"]-a["std_sample"])}. Architecture, environment, training budget, optimizer settings, final evaluation seeds and sampling are matched.{extra} These frozen-policy differences do not establish a general baseline benefit.')
    cp_basic=runs['cartpole_basic']; cp_base=runs['cartpole_baseline']
    cp_train=np.array([float(z['reward']) for z in cp_basic['train']])
    r.paragraph(f'The basic CartPole curve repeatedly reaches the 500-step cap and collapses again. Its last-100 training mean is {number(cp_train[-100:].mean())}, while the final policy evaluates at {number(cp_basic["summary"]["mean"])}. A rolling training window combines trajectories from earlier sampling policies; the final Adam update follows the last logged trajectories and may alter behavior sharply. The baseline final mean {number(cp_base["summary"]["mean"])} is higher for this matched seed, although neither model achieves consistently maximal episode returns.')
    r.paragraph('The strict CartPole discounted objective saturates at 20: a 100-step trajectory has J=19.8816, while 500 steps is almost 20. Thus much of the raw-score difference between long survivals has weak discounted learning signal. This objective/score distinction and noisy episodic updates can coexist with training oscillations; the observations do not isolate a single causal explanation.', 'small')
    pong_rows=[]
    for key in ['pong_basic','pong_baseline']:
        rows=runs[key]['train']
        raw=np.array([float(z['reward']) for z in rows]);disc=np.array([float(z['discounted_return']) for z in rows])
        pong_rows.append([NAMES[key],number(raw[:100].mean()),number(raw[-100:].mean()),number(disc[:100].mean(),4),number(disc[-100:].mean(),4)])
    r.table(['Pong model','First100 raw R','Last100 raw R','First100 J','Last100 J'],pong_rows,[115,95,95,95,95])
    r.paragraph('For Pong, a less negative discounted objective can reflect delaying an early loss while whole-game scores remain near -21. Both policies lost every evaluation game: the basic policy scored between -21 and -17, and the baseline policy between -21 and -14. These results show that the current training configuration did not learn a winning policy.', 'small')
    pong_trace_path=ROOT/'data/validation/pong_trace.json'
    if pong_trace_path.exists():
        pt=read_json(pong_trace_path);pr=pt['results']
        r.paragraph(f'A Pong diagnostic trajectory using the baseline policy and seed {pt["seed"]} lasted {pt["episode_length"]} decisions, with total reward {number(pt["raw_episode_reward"],0)} and discounted J={number(pr["episode_discounted_return"],6)}. Its first nonzero reward was {pr["first_nonzero_reward"]} at zero-based decision {pr["first_nonzero_reward_step"]}, contributing {number(pr["first_nonzero_reward_discounted_contribution"],6)} to J. The reset frame difference had {pr["reset_input_nonzero_pixels"]} nonzero pixels, and a seeded replay reproduced all ten initial image inputs.', 'small')
        r.paragraph('Purpose/theory: verify zero reset history, exact seeded preprocessing, and gamma^t attenuation of the first sparse reward. Method: collect the complete diagnostic game, then replay just ten decisions to compare inputs without adding an evaluation episode. Interpretation: zero early rewards can coexist with nonzero return-to-go from later points. A point remains a reward inside a full game, not a terminal boundary. This trace supports credit-timing semantics rather than convergence.', 'small')
    r.paragraph('Each method/environment was trained with one seed, which limits conclusions about robustness across training runs. Matched evaluation seeds improve comparability but do not remove training-seed uncertainty. Exact trajectory tests establish estimator properties, while the game experiments measure performance under the chosen discounted objective.')
    r.page()

    # Page 13
    r.heading(f'{15 if diagnostic_path.exists() else 14}. Reproducibility and future work')
    r.paragraph('Install the pinned dependencies in requirements-lock.txt, then run the commands below from the project root. The experiment configurations specify the four training budgets, random seeds, and 500-episode evaluation protocol. Per-run CSV files contain rewards, episode lengths, termination flags, step counts, and elapsed times.')
    r.code(['uv venv --python 3.12 .venv',
        'uv pip install --python .venv/bin/python -r requirements-lock.txt',
        '.venv/bin/python src/shuttle.py all',
        '.venv/bin/python scripts/run_experiments.py',
        '.venv/bin/python -m src.validate_rl --output data/validation/rl.json',
        '.venv/bin/python -m src.plot_results --panels',
        '.venv/bin/python -m src.build_report'])
    r.paragraph('Figures and summary statistics are generated from the saved episode records. Checkpoints and configurations support evaluation and resumption at complete update boundaries; the README lists individual training and evaluation commands.')
    r.heading('Given three additional months',2)
    r.paragraph('Run at least 10 matched training seeds per environment/method and report confidence intervals for mean reward differences and sample efficiency. Keep the present 500-episode held-out protocol and reserve separate seeds for tuning. At saved checkpoints, collect independent on-policy batches to estimate gradient covariance directly; compare the implemented EMA with a score-weighted scalar optimum and a learned state-value baseline while holding sample budgets constant.')
    r.paragraph('For shuttle dispatching, vary the hard cap and model overflow as a cost or a deferred queue; compare thresholds, long-run service delays, and out-of-sample arrival laws. Validate against independently implemented tiny exact MDPs and deterministic controls. For policy gradients, verify resumed/uninterrupted equivalence, CPU/MPS gradient agreement on saved batches, and explicit terminal-versus-time-limit cases. Test reward timing and gamma sensitivity without changing the required formal environment.')
    r.paragraph('For Pong, inspect first-score timing, sticky-action effects, frame-difference reset behavior, and sparse-reward credit assignment. Compare image-policy architectures and learning-rate/batch-size grids using tuning seeds. Study sensitivity to the episode-start discounted objective and gamma=.99 in separately defined experimental variants.')
    r.heading('References',2)
    r.paragraph('Assignment specification: mlds_assignment1.pdf.', 'small')
    r.link('Official ALE Pong environment', 'https://ale.farama.org/environments/pong/')
    r.link('Official Gymnasium CartPole environment', 'https://gymnasium.farama.org/environments/classic_control/cart_pole/')
    r.paragraph('Implementation and validation details: docs/shuttle_method.md and docs/rl_validation.md.', 'small')
    return r


def footer(canvas, doc):
    width,height=letter
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#D7E0E4'))
    canvas.line(47,42,width-47,42)
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(GRAY)
    canvas.drawString(47,29,'Yujun Sun | Reinforcement Learning Assignment 1')
    canvas.drawRightString(width-47,29,str(doc.page))
    canvas.restoreState()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-map',type=Path,help='JSON mapping the four canonical names to run folders')
    parser.add_argument('--output',type=Path,default=ROOT/'output/pdf/assignment1_report.pdf')
    parser.add_argument('--check-inputs',action='store_true',help='Validate completeness without generating a final report')
    args=parser.parse_args()
    run_map=read_json(args.run_map) if args.run_map else DEFAULT_RUNS
    if set(run_map)!=set(DEFAULT_RUNS):
        raise ValueError('Run map must contain precisely the four canonical formal runs')
    runs=load_inputs(run_map)
    if args.check_inputs:
        print('All final-report inputs passed: four frozen models, 2,000 full evaluations, eleven plots.')
        return
    report=build_content(runs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    doc=BaseDocTemplate(str(args.output),pagesize=letter,rightMargin=47,leftMargin=47,
        topMargin=43,bottomMargin=54,title='Reinforcement Learning Assignment 1',author='Yujun Sun')
    frame=Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,id='body',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id='Report',frames=[frame],onPage=footer))
    doc.build(report.story)
    (ROOT/'docs/report.md').write_text('\n'.join(report.md)+'\n')
    manifest={key:dict(folder=x['folder'],checkpoint_sha256=x['sha256'],train_episodes=len(x['train']),eval_episodes=len(x['evaluation'])) for key,x in runs.items()}
    (ROOT/'output/pdf/report_manifest.json').write_text(json.dumps(dict(report_sha256=digest(args.output),runs=manifest),indent=2)+'\n')
    print(f'Created {args.output} and docs/report.md from complete saved runs.')


if __name__=='__main__':
    main()
