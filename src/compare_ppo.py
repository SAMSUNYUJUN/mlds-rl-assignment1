"""Rebuild the separate CartPole PPO comparison from complete saved real runs.

This experiment supplements the original assignment's REINFORCE results. It
never trains a model, modifies an original run, or rewrites the original report.
Run only after PPO final.pt is frozen and its 500 full evaluations are saved:
    .venv/bin/python -m src.compare_ppo
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ('cartpole_basic', 'cartpole_baseline', 'cartpole_ppo')
LABELS = {'cartpole_basic': 'REINFORCE, no baseline',
          'cartpole_baseline': 'REINFORCE + prior-batch EMA',
          'cartpole_ppo': 'PPO + value critic / GAE'}
COLORS = {'cartpole_basic': '#247BA0', 'cartpole_baseline': '#218C74',
          'cartpole_ppo': '#C8553D'}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def truth(value):
    if value is True or str(value).lower() in {'true','1'}:
        return True
    if value is False or str(value).lower() in {'false','0'}:
        return False
    raise ValueError(f'Invalid Boolean {value!r}')


def trailing_mean(values, window=100):
    values=np.asarray(values,dtype=np.float64)
    totals=np.r_[0.,np.cumsum(values)]
    end=np.arange(1,len(values)+1);start=np.maximum(0,end-window)
    return (totals[end]-totals[start])/(end-start)


def load_training(path):
    rows=read_csv(path)
    if not rows:
        raise ValueError(f'{path}: no real complete training episodes')
    steps=-1;elapsed=-1.
    for index,row in enumerate(rows):
        if int(row['episode'])!=index:
            raise ValueError(f'{path}: training IDs not contiguous from zero')
        if not (truth(row['terminated']) or truth(row['truncated'])):
            raise ValueError(f'{path}: training row is a rollout slice, not a full episode')
        if int(row['length'])<=0 or int(row['env_steps_total'])<steps:
            raise ValueError(f'{path}: invalid episode length or decreasing step count')
        if not math.isfinite(float(row['reward'])):
            raise ValueError(f'{path}: nonfinite training reward')
        current_elapsed=float(row['elapsed_seconds'])
        if not math.isfinite(current_elapsed) or current_elapsed<elapsed:
            raise ValueError(f'{path}: nonfinite or decreasing training elapsed time')
        expected=(1-.95**int(row['length']))/.05
        if not np.isclose(float(row['discounted_return']),expected,atol=1e-9,rtol=1e-10):
            raise ValueError(f'{path}: full CartPole return differs from the independent geometric formula')
        steps=int(row['env_steps_total']);elapsed=current_elapsed
    return rows


def load_run(root, name, evaluation_file, evaluation_root=None, seed_base=80000000):
    folder=root/'runs'/name
    cfg=read_json(folder/'config.json')
    if cfg.get('game')!='cartpole' or int(cfg['hidden'])!=64 or int(cfg['seed'])!=101 or not np.isclose(float(cfg['gamma']),.95):
        raise ValueError(f'{name}: comparison requires the recorded matched CartPole actor/seed/gamma protocol')
    frozen=read_json(folder/'FROZEN.json')
    file=frozen.get('file','final.pt');sha=digest(folder/file)
    if sha!=frozen['sha256']:
        raise ValueError(f'{name}: frozen checkpoint checksum mismatch')
    training=load_training(folder/'train.csv')
    eval_folder=(root/evaluation_root/name) if evaluation_root else folder
    evaluation=read_csv(eval_folder/evaluation_file)
    protocol_path=eval_folder/'evaluation_protocol.json'
    if evaluation_root:
        protocol=read_json(protocol_path)
        if protocol.get('checkpoint_sha256')!=sha or int(protocol.get('seed_base',-1))!=seed_base or int(protocol.get('episodes',0))!=500 or protocol.get('complete_episodes_only') is not True or protocol.get('action_rule')!='sample categorical policy' or protocol.get('source_run')!=f'runs/{name}':
            raise ValueError(f'{name}: fresh paired evaluation protocol is inconsistent')
    if len(evaluation)!=500:
        raise ValueError(f'{name}: need 500 actual complete evaluations, got {len(evaluation)}')
    train_seeds={int(x['seed']) for x in training}
    eval_seeds=[];evaluation_steps=0;last_elapsed=-1.
    for index,row in enumerate(evaluation):
        if int(row['episode'])!=index or not (truth(row['terminated']) or truth(row['truncated'])):
            raise ValueError(f'{name}: incomplete, duplicated or out-of-order evaluation episode')
        if int(row['length'])<=0 or not math.isfinite(float(row['reward'])):
            raise ValueError(f'{name}: invalid full evaluation episode')
        if row['checkpoint']!=sha:
            raise ValueError(f'{name}: evaluation points to another policy')
        if int(row['seed'])!=seed_base+index:
            raise ValueError(f'{name}: evaluation seed sequence differs from the declared protocol')
        evaluation_steps+=int(row['length'])
        if int(row['env_steps_total'])!=evaluation_steps or not math.isfinite(float(row['elapsed_seconds'])) or float(row['elapsed_seconds'])<last_elapsed:
            raise ValueError(f'{name}: evaluation cumulative steps or elapsed-time counters are inconsistent')
        if not np.isclose(float(row['reward']),int(row['length']),atol=1e-12):
            raise ValueError(f'{name}: CartPole reward must equal complete episode length')
        last_elapsed=float(row['elapsed_seconds']);eval_seeds.append(int(row['seed']))
    if len(set(eval_seeds))!=500 or train_seeds.intersection(eval_seeds):
        raise ValueError(f'{name}: repeated evaluation seeds or overlap with training')
    status=read_json(folder/'status.json')
    if int(frozen['episode'])!=len(training) or int(status['episode'])!=len(training):
        raise ValueError(f'{name}: frozen/status/complete-episode counts disagree')
    for row in training:
        reset_id=int(row.get('trajectory_id',row['episode']))
        if int(row['seed'])!=int(cfg['train_seed_base'])+reset_id or not np.isclose(float(row['reward']),int(row['length']),atol=1e-12):
            raise ValueError(f'{name}: training reset-seed or complete CartPole reward/length mismatch')
    if len(train_seeds)!=len(training):
        raise ValueError(f'{name}: repeated training trajectory seed')
    actual_steps=int(status['steps'])
    completed_steps=sum(int(x['length']) for x in training)
    if actual_steps<completed_steps or actual_steps<int(training[-1]['env_steps_total']):
        raise ValueError(f'{name}: status step count is smaller than recorded interactions')
    if name=='cartpole_ppo':
        if int(cfg['total_steps'])!=actual_steps or int(frozen['steps'])!=actual_steps:
            raise ValueError('PPO did not finish the recorded exact environment-step budget')
        if sum(int(x) for x in status['ongoing_episode_lengths'])!=actual_steps-completed_steps:
            raise ValueError('PPO completed-plus-ongoing episode lengths do not account for every training step')
    saved_summary=read_json(eval_folder/'evaluation_summary.json')
    if int(saved_summary['n'])!=500 or saved_summary['checkpoint_sha256']!=sha or int(saved_summary['eval_seed_base'])!=seed_base:
        raise ValueError(f'{name}: saved evaluation summary protocol mismatch')
    rewards=np.array([float(x['reward']) for x in evaluation])
    if not np.isclose(rewards.mean(),saved_summary['mean']) or not np.isclose(rewards.std(ddof=1),saved_summary['std_sample']) or int(saved_summary['env_steps'])!=evaluation_steps:
        raise ValueError(f'{name}: raw evaluation statistics differ from the saved summary')
    updates=read_csv(folder/'updates.csv')
    if name=='cartpole_ppo':
        if int(updates[-1]['env_steps'])!=actual_steps or sum(int(z['rollout_steps']) for z in updates)!=actual_steps:
            raise ValueError('PPO rollout counters do not equal all recorded actual environment interactions')
        if sum(int(z['actor_gradient_steps']) for z in updates)!=int(status['gradient_steps']):
            raise ValueError('PPO actor minibatch-step counters disagree with final status')
        if sum(int(z['critic_gradient_steps']) for z in updates)!=int(status['gradient_steps']):
            raise ValueError('PPO critic minibatch-step counters disagree with final status')
    train_rewards=np.array([float(x['reward']) for x in training])
    smooth=trailing_mean(train_rewards)
    full_500=np.flatnonzero((np.arange(len(training))>=99) & (smooth==500))
    first_500=int(full_500[0]) if len(full_500) else None
    lengths=np.array([int(x['length']) for x in evaluation])
    both=sum(truth(x['terminated']) and truth(x['truncated']) for x in evaluation)
    terminated=sum(truth(x['terminated']) for x in evaluation)
    truncated=sum(truth(x['truncated']) for x in evaluation)
    stats=dict(run=name,label=LABELS[name],config=cfg,checkpoint_sha256=sha,
               selection_rule=frozen.get('rule','See FROZEN.json'),
               training_complete_episodes=len(training),training_environment_steps=actual_steps,
               training_outer_updates=len(updates),
               training_actor_gradient_steps=int(status.get('gradient_steps',len(updates))),
               training_critic_gradient_steps=sum(int(z.get('critic_gradient_steps',0)) for z in updates),
               training_ongoing_episode_lengths=status.get('ongoing_episode_lengths',[]),
               training_ongoing_episodes=len(status.get('ongoing_episode_lengths',[])),
               training_process_peak_rss_bytes=status.get('process_peak_rss_bytes'),
               steps_in_logged_complete_training_episodes=completed_steps,
               steps_outside_logged_complete_training_episodes=actual_steps-completed_steps,
               train_first100_mean=float(train_rewards[:100].mean()),
               train_last100_mean=float(train_rewards[-100:].mean()),
               train_max_trailing100_mean=float(smooth.max()),
               train_first_full100_mean500_env_steps=int(training[first_500]['env_steps_total']) if first_500 is not None else None,
               train_episodes_after_first_full100_mean500=len(training)-first_500-1 if first_500 is not None else None,
               train_noncap_episodes_after_first_full100_mean500=int(np.sum(train_rewards[first_500+1:]!=500)) if first_500 is not None else None,
               train_elapsed_seconds=float(status.get('elapsed',training[-1]['elapsed_seconds'])),
               evaluation_n=500,evaluation_mean=float(rewards.mean()),
               evaluation_std_sample=float(rewards.std(ddof=1)),evaluation_std_ddof=1,
               evaluation_min=float(rewards.min()),evaluation_max=float(rewards.max()),
               evaluation_mean_length=float(lengths.mean()),evaluation_environment_steps=int(lengths.sum()),
               evaluation_seed_first=eval_seeds[0],evaluation_seed_last=eval_seeds[-1],
               terminated_only=terminated-both,truncated_only=truncated-both,
               both_terminated_and_truncated=both,
               evaluation_cap_hits=int(np.sum(lengths==500)),
               source_files=dict(training=f'runs/{name}/train.csv',evaluation=str((eval_folder/evaluation_file).relative_to(root)),
                                 configuration=f'runs/{name}/config.json',frozen=f'runs/{name}/FROZEN.json'))
    return dict(stats=stats,training=training,evaluation=evaluation,
                rewards=rewards,seeds=eval_seeds)


def paired_summary(ppo, comparator, rng_seed, bootstrap_samples):
    if ppo['seeds']!=comparator['seeds']:
        raise ValueError('Paired comparisons require identical ordered evaluation seed IDs')
    differences=ppo['rewards']-comparator['rewards']
    rng=np.random.default_rng(rng_seed)
    means=[]
    for start in range(0,bootstrap_samples,200):
        size=min(200,bootstrap_samples-start)
        indices=rng.integers(0,len(differences),size=(size,len(differences)))
        means.extend(differences[indices].mean(axis=1).tolist())
    low,high=np.quantile(means,[.025,.975])
    return dict(reference=comparator['stats']['run'],difference='PPO raw reward minus reference raw reward',
                n=500,mean=float(differences.mean()),std_sample=float(differences.std(ddof=1)),
                standard_error=float(differences.std(ddof=1)/np.sqrt(len(differences))),
                positive=int(np.sum(differences>0)),equal=int(np.sum(differences==0)),
                negative=int(np.sum(differences<0)),
                bootstrap_percentile_95_ci=[float(low),float(high)],
                bootstrap_samples=bootstrap_samples,bootstrap_seed=rng_seed,
                inference_scope='Conditional on these fixed checkpoints and matched evaluation seeds; excludes training-seed uncertainty'),differences


def setup_plots():
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'mlds-ppo-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'savefig.dpi':180,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.dpi':120})
    return plt


def plots(root, loaded, paired):
    plt=setup_plots();folder=root/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(11.6,5.5),constrained_layout=True)
    for name in RUNS:
        rows=loaded[name]['training'];r=np.array([float(x['reward']) for x in rows]);steps=[int(x['env_steps_total']) for x in rows]
        ax.plot(steps,r,color=COLORS[name],alpha=.12,linewidth=.45)
        ax.plot(steps,trailing_mean(r),color=COLORS[name],linewidth=1.8,label=LABELS[name])
    ax.set(xlabel='Actual training environment interactions',ylabel='Raw complete-episode total reward',
           title='CartPole-v1: saved training rewards and trailing 100-episode means',ylim=(0,510))
    ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=9)
    ax.ticklabel_format(axis='x',style='sci',scilimits=(6,6))
    fig.savefig(folder/'ppo_comparison_training.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.4),sharex=True,sharey=True,constrained_layout=True)
    for ax,name in zip(axes,RUNS):
        r=loaded[name]['rewards'];stat=loaded[name]['stats']
        ax.hist(r,bins=np.linspace(0,500,26),color=COLORS[name],alpha=.8,edgecolor='white')
        ax.axvline(r.mean(),color='#222222',linestyle='--',linewidth=1)
        ax.set(title=LABELS[name],xlabel='Raw full-episode reward',xlim=(0,505))
        ax.text(.03,.97,f'n=500\nMean={r.mean():.3f}\nSample SD={r.std(ddof=1):.3f}',
                transform=ax.transAxes,va='top',fontsize=9,
                bbox=dict(facecolor='white',alpha=.92,edgecolor='#DDDDDD'))
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Number of complete evaluation episodes')
    fig.suptitle('Frozen checkpoints: identical held-out seeds and stochastic action sampling')
    fig.savefig(folder/'ppo_comparison_evaluation.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.3),sharex=True,constrained_layout=True)
    for ax,name in zip(axes,RUNS[:2]):
        stats,diff=paired[name]
        ax.hist(diff,bins=np.arange(-500,521,20),color=COLORS[name],alpha=.8,edgecolor='white')
        ax.axvline(0,color='#555555',linewidth=1);ax.axvline(diff.mean(),color=COLORS['cartpole_ppo'],linestyle='--')
        ax.set(title='PPO minus '+LABELS[name],xlabel='Paired full-episode reward difference',ylabel='Episode pairs')
        ax.text(.03,.97,f'Mean={stats["mean"]:.3f}\n95% bootstrap CI\n[{stats["bootstrap_percentile_95_ci"][0]:.3f}, {stats["bootstrap_percentile_95_ci"][1]:.3f}]',
                transform=ax.transAxes,va='top',fontsize=9,
                bbox=dict(facecolor='white',alpha=.92,edgecolor='#DDDDDD'))
        ax.grid(axis='y',alpha=.2)
    fig.savefig(folder/'ppo_comparison_paired_differences.png');plt.close(fig)


def write_discussion(root, loaded, paired, summary):
    ppo=loaded['cartpole_ppo']['stats']
    basic=loaded['cartpole_basic']['stats']
    ema=loaded['cartpole_baseline']['stats']
    cfg=ppo['config'];bootstrap_n=paired[RUNS[0]][0]['bootstrap_samples']
    fresh_first=summary['primary_evaluation_seed_base']
    validation=read_json(root/'data/validation/ppo.json')
    if validation['status']!='passed':
        raise ValueError('Saved PPO validation has not passed')
    ratio_basic=ppo['train_elapsed_seconds']/basic['train_elapsed_seconds']
    ratio_ema=ppo['train_elapsed_seconds']/ema['train_elapsed_seconds']
    lines=['# CartPole: PPO comparison with REINFORCE','',
        'We compare PPO-Clip with basic REINFORCE and REINFORCE with a prior-batch EMA baseline on CartPole-v1.','',
        '## Design and method','',
        f'PPO and basic REINFORCE each use {ppo["training_environment_steps"]:,} environment interactions. The EMA reference uses {ema["training_environment_steps"]:,}; all three methods therefore do not have identical sample budgets. Each method has one training seed, 101. The final update at the fixed budget is frozen before evaluation.',
        'The actor is 4 inputs -> 64 Tanh units -> 2 action logits, initialized with the same PyTorch seed before constructing any critic. PPO adds a separate 4-64-Tanh-1 state-value critic. All methods use the native CPU CartPole-v1 environment, gamma=.95, original rewards and the official 500-step limit. Actions are sampled rather than chosen by argmax.','',
        'PPO uses the following estimators and losses. Old values and log probabilities are frozen at rollout collection. At a natural terminal, next-state value is suppressed. At a TimeLimit, the final observation is bootstrapped but GAE carry stops before the reset episode. A rollout slice bootstraps its last next-state value without becoming an episode termination.','',
        '```text',
        'delta_t = r_t + gamma * (1-terminated_t) * V_old(s_next) - V_old(s_t)',
        'A_raw_t = delta_t + gamma * lambda * (1-(terminated_t or truncated_t)) * A_raw_(t+1)',
        'value_target_t = detach(A_raw_t + V_old(s_t))',
        'A_t = detach((A_raw_t - rollout_mean(A_raw)) / (rollout_population_SD(A_raw) + 1e-8))',
        'ratio_t = exp(log pi_new(a_t|s_t) - log pi_old(a_t|s_t))',
        'actor_loss = -mean(min(ratio_t*A_t, clip(ratio_t,1-epsilon,1+epsilon)*A_t))',
        'critic_loss = 0.5 * mean((V_new(s_t) - value_target_t)^2)',
        '```','',
        f'Here lambda={cfg["gae_lambda"]}, epsilon={cfg["clip_ratio"]}, actor Adam LR={cfg["actor_lr"]}, and critic Adam LR={cfg["critic_lr"]}. Each {cfg["rollout_steps"]}-transition rollout across {cfg["num_envs"]} lanes is reused for {cfg["epochs"]} epochs in minibatches of {cfg["minibatch_size"]}. Actor and critic gradient norms are clipped at {cfg["grad_clip"]}. Entropy coefficient is {cfg["entropy_coef"]}; observation and reward normalization are disabled. The actor surrogate uniformly averages transitions and has no absolute episode-start gamma^t factor.','',
        '## Primary fresh paired evaluation','',
        f'Each frozen model runs 500 complete episodes on the same new seeds {fresh_first} through {fresh_first+499}. This set is separate from training, tuning and the historical final evaluation. Per-episode action-sampling uniforms follow the same seeded protocol.','',
        '| Method | Full train episodes | Actual train steps | Last 100 train mean | Fresh eval mean | Sample SD | Range |',
        '|---|---:|---:|---:|---:|---:|---|']
    for name in RUNS:
        stat=loaded[name]['stats']
        lines.append(f'| {LABELS[name]} | {stat["training_complete_episodes"]:,} | {stat["training_environment_steps"]:,} | {stat["train_last100_mean"]:.3f} | {stat["evaluation_mean"]:.3f} | {stat["evaluation_std_sample"]:.3f} | [{stat["evaluation_min"]:.0f}, {stat["evaluation_max"]:.0f}] |')
    lines += ['',f'PPO reaches the 500-step cap in {ppo["evaluation_cap_hits"]}/500 fresh episodes: observed mean {ppo["evaluation_mean"]:.3f} and sample SD {ppo["evaluation_std_sample"]:.3f}. Basic REINFORCE reaches the cap in {basic["evaluation_cap_hits"]}/500; EMA REINFORCE in {ema["evaluation_cap_hits"]}/500. These bounded observations do not assess survival beyond 500 steps. Standard deviations use ddof=1; termination and truncation can coincide on step 500.','',
        '![Raw training rewards and trailing 100-episode means](../figures/ppo_comparison_training.png)','',
        'Figure 1. Faint lines preserve full-episode rewards; solid lines show the trailing 100 complete-episode mean, with a growing window before episode 100. The horizontal axis is actual environment interactions. Equal 100-episode windows span different numbers of transitions. PPO training episodes can cross several policy updates, so their scores are not evaluations of one stationary policy.','',
        f'The PPO curve first attains a full 100-episode mean of 500 at {ppo["train_first_full100_mean500_env_steps"]:,} interactions. All {ppo["train_episodes_after_first_full100_mean500"]:,} subsequently completed training episodes also score 500. Basic REINFORCE shows repeated high-score windows and collapses; its last-100 training mean {basic["train_last100_mean"]:.3f} does not describe its final frozen evaluation mean {basic["evaluation_mean"]:.3f}. Earlier checkpoints are not substituted.',
        f'PPO logs {ppo["training_complete_episodes"]:,} completed training episodes and ends with {ppo["training_ongoing_episodes"]} ongoing episodes containing {ppo["steps_outside_logged_complete_training_episodes"]:,} transitions. Completed plus ongoing lengths account for every one of the {ppo["training_environment_steps"]:,} interactions. Those unfinished transitions contribute bootstrapped GAE updates but are not counted as full episodes.','',
        '![Fresh frozen-policy full-episode histograms](../figures/ppo_comparison_evaluation.png)','',
        'Figure 2. The panels share axes and bins. All 500 PPO rewards lie at the official cap; the two REINFORCE distributions remain dispersed. Each count is one complete natural-termination or official-TimeLimit episode. The dashed line marks the mean.','',
        '## Optimization work and elapsed time','',
        '| Method | Rollout/batch updates | Actor optimizer steps | Critic optimizer steps | Recorded training seconds |',
        '|---|---:|---:|---:|---:|']
    for name in RUNS:
        stat=loaded[name]['stats']
        lines.append(f'| {LABELS[name]} | {stat["training_outer_updates"]:,} | {stat["training_actor_gradient_steps"]:,} | {stat["training_critic_gradient_steps"]:,} | {stat["train_elapsed_seconds"]:.3f} |')
    lines += ['',f'PPO uses {ppo["training_outer_updates"]} rollouts, {ppo["training_actor_gradient_steps"]:,} actor steps and {ppo["training_critic_gradient_steps"]:,} critic steps. Its recorded {ppo["train_elapsed_seconds"]:.3f} seconds are {ratio_basic:.2f} times the basic run and {ratio_ema:.2f} times the EMA run. Equal interactions do not imply equal computation. These are single recorded training timings, not a repeated controlled speed benchmark.','',
        '## Paired differences between fixed policies','',
        '| PPO minus reference | Pairs | Mean reward difference | Difference sample SD | 95% paired-bootstrap interval | Positive / equal / negative |',
        '|---|---:|---:|---:|---|---|']
    for name in RUNS[:2]:
        stat,_=paired[name];lo,hi=stat['bootstrap_percentile_95_ci']
        lines.append(f'| {LABELS[name]} | 500 | {stat["mean"]:.3f} | {stat["std_sample"]:.3f} | [{lo:.3f}, {hi:.3f}] | {stat["positive"]} / {stat["equal"]} / {stat["negative"]} |')
    lines += ['', '![Paired reward difference distributions](../figures/ppo_comparison_paired_differences.png)','',
        f'Figure 3. PPO has no negative reward difference in these pairs; ties occur when a reference also reaches 500. Percentile intervals resample the 500 reset-seed/action-uniform pairs {bootstrap_n:,} times using recorded fixed bootstrap seeds. They condition on these frozen policies and exclude training-seed uncertainty.','',
        '## Historical evaluation: secondary context','',
        'The separate historical seed range remains 80000000-80000499. These results are not pooled with the fresh comparison or used for checkpoint selection.','',
        '| Method | Historical episodes | Mean reward | Sample SD |',
        '|---|---:|---:|---:|']
    for name in RUNS:
        hist=summary['historical_evaluations'][name]
        lines.append(f'| {LABELS[name]} | 500 | {hist["evaluation_mean"]:.3f} | {hist["evaluation_std_sample"]:.3f} |')
    resume=validation['checkpoint_resume'];first=validation['first_update_and_critic']
    gae_error=max(case['max_absolute_error'] for case in validation['gae']['cases'])
    lines += ['', '## Algorithm validation','',
        f'Matched actor initialization has maximum parameter error {validation["initialization"]["actor_parameter_max_absolute_error"]:.1f}. All {len(validation["gae"]["cases"])} hand-derived GAE boundary cases match with maximum error {gae_error:.1f}, including natural terminals, TimeLimits, rollout ends and reset-carry blocking. Six signed-advantage clipping cases match the expected loss and derivative.',
        f'The first production actor update matches an independent unclipped on-policy calculation with parameter error {first["first_update_actor_parameter_max_absolute_error"]:.1f}; advantages and critic targets receive no gradients. Fixed-target critic MSE falls from {first["critic_fixed_target_mse_before"]:.6f} to {first["critic_fixed_target_mse_after"]:.6f}.',
        f'Checkpoint restoration reproduces actual next-rollout tensors, episode records, both Adam states and actor/critic parameters exactly. The boundary test uses rollouts {resume["rollout_lengths"]} for exactly {resume["final_environment_steps"]} steps: {resume["total_completed_episode_steps"]} completed-episode steps plus {sum(resume["final_ongoing_episode_lengths"])} ongoing steps. These tests validate mechanics, not general performance superiority.','',
        '## Fairness and limits','',
        '- PPO changes a bundle: state-value critic, GAE, lower actor learning rate, minibatch reuse, centered/scaled advantages, gradient clipping and uniform transition weighting. Original REINFORCE uses episode-start gamma^t weighting, Monte Carlo returns, Adam LR 0.01, one update per eight full episodes, a fixed 100 loss scale and gradient bound 10. This is not a one-component clipping ablation.',
        '- PPO bootstraps TimeLimits from a critic. Episodic REINFORCE ends its observed discounted objective at the official limit. Therefore estimator target definitions also differ.',
        '- There is one matched training seed; the EMA reference has fewer realized interactions. Evaluation reward SD is distinct from policy-gradient covariance. Bootstrap intervals for fixed policies do not give across-training-seed confidence intervals.',
        '- All PPO scores are capped at 500 in both evaluation sets. The observed zero SD does not establish zero uncertainty on unseen initializations or longer horizons. Repeated training seeds and matched component ablations are needed for causal algorithm claims.','',
        '## 中文学习解释','',
        f'新的配对评估使用 {fresh_first} 起的 500 个种子，PPO 均分 {ppo["evaluation_mean"]:.3f}、样本标准差 {ppo["evaluation_std_sample"]:.3f}，500/500 回合达到官方上限。基础版和 EMA 版的新评估均分分别为 {basic["evaluation_mean"]:.3f} 和 {ema["evaluation_mean"]:.3f}。原先 80000000 起的结果单列，未混合或用于挑选 checkpoint。',
        f'PPO 与基础版环境步数相同，但用了 {ppo["training_outer_updates"]} 次 rollout 更新、actor 和 critic 各 {ppo["training_actor_gradient_steps"]:,} 次优化，记录训练耗时 {ppo["train_elapsed_seconds"]:.2f} 秒，约为基础版的 {ratio_basic:.2f} 倍。它同时改变 critic、GAE、学习率、复用、标准化和时间权重，不能把表现差异只归因于 clipping。',
        '第一张图显示 PPO 的训练表现保持在上限，基础版多次高分后崩落；第二张图是相同新种子的完整评估；第三张图给出配对奖励差。训练未完成片段可用于 bootstrap，但不计为完整回合。单个训练 seed 和固定策略的 bootstrap 不能证明所有训练条件下 PPO 都更好。','',
        '## Commands and retained evidence','',
        '```sh',
        '.venv/bin/python src/ppo.py train --run runs/cartpole_ppo --steps 1384029 --seed 101',
        '.venv/bin/python src/ppo.py evaluate --run runs/cartpole_ppo --episodes 500 --eval-seed-base 80000000',
        'for run_name in cartpole_basic cartpole_baseline cartpole_ppo; do',
        '  .venv/bin/python src/ppo.py evaluate --run "runs/${run_name}" --output "runs/ppo_comparison_eval/${run_name}" --episodes 500 --eval-seed-base 84000000',
        'done',
        '.venv/bin/python -m src.validate_ppo',
        '.venv/bin/python -m src.compare_ppo',
        '```','',
        'The train/evaluate commands preserve frozen or already completed runs. The last command rebuilds the figures and this document without training. Full effective configurations, checkpoint hashes, counters, fresh/historical statistics and bootstrap seeds are retained in [comparison metrics](../data/ppo_comparison_summary.json). Raw logs and protocol files remain under the corresponding run folders.',
        'Quantitative validation: [PPO validation](../data/validation/ppo.json). Additional counter, clipping and original-artifact hash checks: [PPO integrity check](../data/ppo_integrity_check.json).']
    (root/'docs/PPO_COMPARISON.md').write_text('\n'.join(lines)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--evaluation-file',default='evaluation.csv')
    parser.add_argument('--evaluation-root',default='runs/ppo_comparison_eval')
    parser.add_argument('--seed-base',type=int,default=84000000)
    parser.add_argument('--bootstrap-samples',type=int,default=2000)
    parser.add_argument('--bootstrap-seed',type=int,default=1012026)
    args=parser.parse_args();root=args.root.resolve()
    if args.bootstrap_samples<100:
        parser.error('At least 100 bootstrap samples are required for the descriptive interval')
    # Read/validate all genuine data before generating any comparison artifacts.
    loaded={name:load_run(root,name,args.evaluation_file,args.evaluation_root,args.seed_base) for name in RUNS}
    historical={name:load_run(root,name,args.evaluation_file,None,80000000) for name in RUNS}
    expected=[args.seed_base+i for i in range(500)]
    if any(item['seeds']!=expected for item in loaded.values()):
        raise ValueError('Fresh paired comparison requires its declared seed base and exactly 500 ordered full episodes')
    paired={name:paired_summary(loaded['cartpole_ppo'],loaded[name],args.bootstrap_seed+i,args.bootstrap_samples)
            for i,name in enumerate(RUNS[:2])}
    steps={name:loaded[name]['stats']['training_environment_steps'] for name in RUNS}
    summary=dict(complete=True,comparison='Additional PPO experiment versus original frozen CartPole REINFORCE policies',
        runs={name:loaded[name]['stats'] for name in RUNS},
        primary_evaluation_seed_base=args.seed_base,
        primary_evaluation_directory=args.evaluation_root,
        historical_evaluation_seed_base=80000000,
        historical_evaluations={name:historical[name]['stats'] for name in RUNS},
        ppo_training_environment_steps=steps['cartpole_ppo'],basic_training_environment_steps=steps['cartpole_basic'],
        ema_training_environment_steps=steps['cartpole_baseline'],
        ppo_matches_basic_environment_steps=steps['cartpole_ppo']==steps['cartpole_basic'],
        all_three_environment_step_budgets_equal=len(set(steps.values()))==1,
        evaluation_action_rule='Stochastic categorical policy sampling; identical ordered held-out episode seeds',
        paired_differences={name:pair[0] for name,pair in paired.items()},
        limitations=['One training seed per method','EMA REINFORCE has a different actual environment-step budget',
            'PPO changes critic,GAE,learning rate,minibatch reuse and uniform transition weighting together',
            'Paired-bootstrap intervals condition on frozen checkpoints,not training-seed variation'])
    plots(root,loaded,paired)
    (root/'data/ppo_comparison_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    write_discussion(root,loaded,paired,summary)
    print(json.dumps(dict(complete=True,evaluation_episodes={name:500 for name in RUNS},training_steps=steps,
                         output='data/ppo_comparison_summary.json'),indent=2))


if __name__=='__main__':
    main()
