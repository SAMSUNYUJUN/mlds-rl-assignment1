"""Rebuild the one-step TD critic-target ablation from frozen saved data.

Actor GAE is unchanged. No training or original-report editing occurs here.
Run after both 500-episode evaluations and initial-value diagnostics finish:
    .venv/bin/python -m src.compare_td1_critic
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,tempfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUNS={'gae':'cartpole_ppo','td1':'cartpole_ppo_td1_critic'}
LABELS={'gae':'GAE-derived critic target','td1':'One-step TD critic target'}
COLORS={'gae':'#247BA0','td1':'#D05A3C'}


def read_json(path):return json.loads(Path(path).read_text())
def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def truth(x):
    if str(x).lower() in {'true','1'}:return True
    if str(x).lower() in {'false','0'}:return False
    raise ValueError(f'Invalid Boolean: {x!r}')
def smooth(values,window=100):
    x=np.asarray(values,np.float64);total=np.r_[0.,np.cumsum(x)]
    end=np.arange(1,len(x)+1);start=np.maximum(0,end-window)
    return (total[end]-total[start])/(end-start)
def check(condition,message):
    if not condition:raise ValueError(message)


def load_arm(root,arm):
    name=RUNS[arm];folder=root/'runs'/name;cfg=read_json(folder/'config.json')
    frozen=read_json(folder/'FROZEN.json');sha=digest(folder/frozen['file'])
    check(sha==frozen['sha256'],f'{name}: frozen hash mismatch')
    check(cfg['game']=='cartpole' and cfg['seed']==101 and cfg['device']=='cpu',f'{name}: unexpected local experiment protocol')
    check(cfg.get('critic_target','gae')==arm,f'{name}: wrong critic-target label')
    train=read_csv(folder/'train.csv');updates=read_csv(folder/'updates.csv');status=read_json(folder/'status.json')
    check(bool(train) and bool(updates),f'{name}: incomplete training logs')
    last_step=0;last_time=0.;complete_steps=0
    for i,row in enumerate(train):
        length=int(row['length']);step=int(row['env_steps_total']);elapsed=float(row['elapsed_seconds'])
        check(int(row['episode'])==i and (truth(row['terminated']) or truth(row['truncated'])),f'{name}: a training row is not a complete episode')
        check(0<length<=500 and np.isclose(float(row['reward']),length),f'{name}: CartPole reward/length mismatch')
        check(step>=last_step and elapsed>=last_time and math.isfinite(elapsed),f'{name}: invalid training counters')
        check(int(row['seed'])==cfg['train_seed_base']+int(row['trajectory_id']),f'{name}: reset seed mismatch')
        check(np.isclose(float(row['discounted_return']),(1-cfg['gamma']**length)/(1-cfg['gamma']),atol=1e-9),f'{name}: discounted return mismatch')
        complete_steps+=length;last_step=step;last_time=elapsed
    steps=int(status['steps']);ongoing=[int(x) for x in status['ongoing_episode_lengths']]
    check(steps==cfg['total_steps']==frozen['steps'],f'{name}: final budget mismatch')
    check(len(train)==status['episode']==frozen['episode'],f'{name}: completed-episode count mismatch')
    check(complete_steps+sum(ongoing)==steps,f'{name}: completed/ongoing accounting mismatch')
    cumulative=0
    for i,row in enumerate(updates):
        cumulative+=int(row['rollout_steps'])
        check(int(row['update'])==i+1 and int(row['env_steps'])==cumulative,f'{name}: rollout counter mismatch')
        check(float(row['value_loss'])>=0 and math.isfinite(float(row['value_loss'])),f'{name}: invalid half-MSE log')
    actor_steps=sum(int(x['actor_gradient_steps']) for x in updates)
    critic_steps=sum(int(x['critic_gradient_steps']) for x in updates)
    check(cumulative==steps and actor_steps==critic_steps==status['gradient_steps'],f'{name}: optimization-step counter mismatch')
    eval_folder=root/'runs/td1_critic_eval'/arm
    protocol=read_json(eval_folder/'evaluation_protocol.json')
    check(protocol['source_run']==f'runs/{name}' and protocol['checkpoint_sha256']==sha and protocol['seed_base']==88000000 and protocol['episodes']==500 and protocol['complete_episodes_only'] is True and protocol['action_rule']=='sample categorical policy',f'{name}: fresh evaluation protocol mismatch')
    evaluation=read_csv(eval_folder/'evaluation.csv');check(len(evaluation)==500,f'{name}: need 500 complete fresh episodes')
    eval_steps=0
    for i,row in enumerate(evaluation):
        check(int(row['episode'])==i and int(row['seed'])==88000000+i and row['checkpoint']==sha,f'{name}: fresh seed/hash mismatch')
        check(truth(row['terminated']) or truth(row['truncated']),f'{name}: incomplete evaluation episode')
        check(0<int(row['length'])<=500 and float(row['reward'])==int(row['length']),f'{name}: invalid evaluation length/reward')
        eval_steps+=int(row['length']);check(int(row['env_steps_total'])==eval_steps,f'{name}: evaluation step counter mismatch')
    r=np.array([float(x['reward']) for x in evaluation]);tr=np.array([float(x['reward']) for x in train]);ma=smooth(tr)
    saved=read_json(eval_folder/'evaluation_summary.json')
    check(saved['n']==500 and saved['checkpoint_sha256']==sha and np.isclose(saved['mean'],r.mean()) and np.isclose(saved['std_sample'],r.std(ddof=1)),f'{name}: saved statistics mismatch')
    diagnostic=read_csv(eval_folder/'initial_value_diagnostic.csv');check(len(diagnostic)==500,f'{name}: need 500 initial-value predictions')
    states=[];values=[];returns=[];errors=[]
    for i,row in enumerate(diagnostic):
        v=float(row['value_prediction']);g=float(row['mc_return']);e=float(row['error'])
        check(int(row['episode'])==i and int(row['seed'])==88000000+i and row['checkpoint']==sha,f'{name}: initial-value diagnostic protocol mismatch')
        check(math.isfinite(v) and math.isfinite(g) and np.isclose(g,float(evaluation[i]['discounted_return']),atol=1e-9) and np.isclose(e,v-g,atol=1e-10),f'{name}: initial-value diagnostic calculation mismatch')
        states.append([float(row[f'state{j}']) for j in range(4)]);values.append(v);returns.append(g);errors.append(e)
    errors=np.array(errors);diag=dict(n=500,mean_prediction=float(np.mean(values)),mean_mc_return=float(np.mean(returns)),bias=float(errors.mean()),rmse=float(np.sqrt(np.mean(errors**2))),mae=float(np.abs(errors).mean()),error_std_sample=float(errors.std(ddof=1)))
    summary_path=next((eval_folder/x for x in ['initial_value_diagnostic_summary.json','initial_value_diagnostic.json'] if (eval_folder/x).exists()),None)
    check(summary_path is not None,f'{name}: initial-value diagnostic summary missing')
    ds=read_json(summary_path);ds=ds.get('summary',ds)
    check(ds['checkpoint_sha256']==sha and ds['evaluation_csv_sha256']==digest(eval_folder/'evaluation.csv'),f'{name}: stale diagnostic source hashes')
    aliases={'mean_prediction':'mean_pred','mean_mc_return':'mean_return','error_std_sample':'sample_sd_error'}
    for key in diag:
        saved_key=key if key in ds else aliases.get(key,key)
        check(np.isclose(ds[saved_key],diag[key],atol=1e-10),f'{name}: diagnostic summary {key} mismatch')
    crossings={}
    for level in [475,500]:
        found=np.flatnonzero((np.arange(len(ma))>=99)&(ma>=level))
        crossings[str(level)]=int(train[int(found[0])]['env_steps_total']) if len(found) else None
    full500=np.flatnonzero((np.arange(len(ma))>=99)&(ma==500))
    first500=int(full500[0]) if len(full500) else None
    tail500=0
    for reward in tr[::-1]:
        if reward!=500:break
        tail500+=1
    stability=dict(first_full500_window_episode=first500,
                   subsequent_complete_episodes=len(tr)-first500-1 if first500 is not None else None,
                   subsequent_noncap_episodes=int(np.sum(tr[first500+1:]<500)) if first500 is not None else None,
                   all_subsequent_complete_episodes_score500=bool(np.all(tr[first500+1:]==500)) if first500 is not None else None,
                   subsequent_minimum_trailing100_mean=float(ma[first500:].min()) if first500 is not None else None,
                   consecutive_cap_episodes_at_end=tail500)
    both=sum(truth(x['terminated']) and truth(x['truncated']) for x in evaluation)
    stats=dict(run=name,configuration=cfg,checkpoint_sha256=sha,training_steps=steps,complete_training_episodes=len(train),ongoing_episode_lengths=ongoing,ongoing_training_steps=sum(ongoing),rollout_updates=len(updates),actor_optimizer_steps=actor_steps,critic_optimizer_steps=critic_steps,training_elapsed_seconds=float(status['elapsed']),train_first100_mean=float(tr[:100].mean()),train_last100_mean=float(tr[-100:].mean()),first_full100_window_crossing_steps=crossings,evaluation_n=500,evaluation_mean=float(r.mean()),evaluation_std_sample=float(r.std(ddof=1)),evaluation_min=float(r.min()),evaluation_max=float(r.max()),evaluation_cap_hits=int(np.sum(r==500)),evaluation_environment_steps=eval_steps,terminated_only=sum(truth(x['terminated']) for x in evaluation)-both,truncated_only=sum(truth(x['truncated']) for x in evaluation)-both,both_terminated_and_truncated=both,initial_value_diagnostic=diag,training_mse_definition='2 * logged value_loss: unweighted mean of pre-update minibatch MSE observations across reuse epochs; targets differ between arms')
    stats['training_stability_after_first_full500_window']=stability
    stats['training_target_mse_first_update']=2*float(updates[0]['value_loss'])
    stats['training_target_mse_last_update']=2*float(updates[-1]['value_loss'])
    return dict(stats=stats,train=train,updates=updates,evaluation=evaluation,rewards=r,states=np.array(states))


def make_plots(root,arms):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'mlds-td1-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
    folder=root/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(13,4.9),sharey=True,constrained_layout=True)
    for arm,data in arms.items():
        rows=data['train'];x=[int(z['env_steps_total']) for z in rows];y=[float(z['reward']) for z in rows]
        for ax in axes:
            ax.plot(x,y,color=COLORS[arm],alpha=.17,linewidth=.45)
            ax.plot(x,smooth(y),color=COLORS[arm],linewidth=1.8,label=LABELS[arm])
    axes[0].set(title='Full matched interaction budget',xlim=(0,arms['gae']['stats']['training_steps']))
    axes[1].set(title='First 400,000 interactions',xlim=(0,400000))
    for arm,data in arms.items():
        crossing=data['stats']['first_full100_window_crossing_steps']['500']
        if crossing is not None and crossing<=400000:
            axes[1].scatter([crossing],[500],s=24,color=COLORS[arm],zorder=5)
            axes[1].annotate(f'{crossing:,} steps\n100-episode mean = 500',
                             xy=(crossing,500),xytext=(crossing,455 if arm=='gae' else 415),
                             ha='center',va='top',fontsize=8,color=COLORS[arm],
                             bbox=dict(facecolor='white',alpha=.92,edgecolor='none',pad=2),
                             arrowprops=dict(arrowstyle='->',color=COLORS[arm],linewidth=.8))
    for ax in axes:
        ax.set(xlabel='Actual environment interactions',ylim=(0,510));ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
        ax.ticklabel_format(axis='x',style='sci',scilimits=(3,3))
    axes[0].set_ylabel('Raw full-episode reward; trailing 100 mean')
    fig.suptitle('Only the critic label changes; actor advantages remain GAE')
    fig.savefig(folder/'td1_critic_training.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.7,4.8),sharex=True,sharey=True,constrained_layout=True)
    for ax,(arm,data) in zip(axes,arms.items()):
        x=[int(z['env_steps']) for z in data['updates']];y=np.array([2*float(z['value_loss']) for z in data['updates']])
        ax.plot(x,np.ma.masked_less_equal(y,0),color=COLORS[arm],alpha=.35,linewidth=.7,label='Logged minibatch target MSE')
        ax.plot(x,np.ma.masked_less_equal(smooth(y,20),0),color=COLORS[arm],linewidth=1.6,label='Trailing 20-update mean')
        ax.set(title=LABELS[arm],xlabel='Actual environment interactions',yscale='log');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Training target MSE (different labels)')
    fig.suptitle('Own-target fitting losses: lower does not mean more accurate value predictions')
    fig.savefig(folder/'td1_critic_training_mse.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.3,4.6),sharex=True,sharey=True,constrained_layout=True)
    for ax,(arm,data) in zip(axes,arms.items()):
        r=data['rewards'];ax.hist(r,bins=np.linspace(0,500,26),color=COLORS[arm],edgecolor='white',alpha=.85)
        ax.axvline(r.mean(),color='#333333',linestyle='--',linewidth=1)
        ax.text(.03,.97,f'n = 500\nMean = {r.mean():.3f}\nSample SD = {r.std(ddof=1):.3f}',transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='#DDDDDD'))
        ax.set(title=LABELS[arm],xlabel='Raw complete-episode reward',xlim=(0,505));ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Complete evaluation episodes')
    fig.suptitle('Frozen policies: 500 shared fresh seeds, stochastic actions')
    fig.savefig(folder/'td1_critic_evaluation.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.7,4.5),constrained_layout=True)
    for ax,metric,title in zip(axes,['bias','rmse'],['Mean error: V(s0) - observed G0','RMSE versus one observed return per state']):
        heights=[arms[a]['stats']['initial_value_diagnostic'][metric] for a in RUNS]
        bars=ax.bar([0,1],heights,color=[COLORS[a] for a in RUNS],width=.6)
        ax.set(xticks=[0,1],xticklabels=['GAE label','One-step TD label'],title=title,ylabel='Discounted-return units')
        ax.axhline(0,color='#555555',linewidth=.7);ax.grid(axis='y',alpha=.2)
        ax.bar_label(bars,labels=[f'{v:.5g}' for v in heights],padding=4,fontsize=9)
        ax.margins(y=.2)
    fig.suptitle('Initial-state prediction diagnostic under each frozen policy (500 shared states)')
    fig.savefig(folder/'td1_critic_initial_value.png');plt.close(fig)


def write_doc(root,arms,summary):
    a=arms['gae']['stats'];b=arms['td1']['stats']
    validation=read_json(root/'data/validation/td1_critic.json')
    lines=['# One-step TD critic target: a PPO ablation','',
        'We change only the critic regression label. The actor still uses normalized GAE advantages, gamma=.95 and lambda=.95; the network, initialization, optimizers, data reuse and interaction budget are held fixed.','',
        'In this seed, the GAE-derived critic label reaches high training rewards sooner. Both frozen final policies nevertheless reach 500 in all 500 fresh evaluations; the official score cap makes their final observed rewards tie.','',
        '## What changes','',
        '```text',
        'delta_t = r_t + gamma * (1-terminated_t) * V_old(s_next) - V_old(s_t)',
        'A_GAE_t = delta_t + gamma*lambda*(1-(terminated_t or truncated_t))*A_GAE_(t+1)',
        'GAE critic label:      y_t = V_old(s_t) + A_GAE_t',
        'One-step TD label:     y_t = r_t + gamma*(1-terminated_t)*V_old(s_next)',
        'Both critic losses:    0.5 * mean((V_new(s_t) - detach(y_t))^2)',
        'Both actor advantages: detach(normalize(A_GAE_t))',
        '```','',
        'Both labels use the frozen rollout critic throughout all reuse epochs. A TimeLimit bootstraps the final observation; a natural terminal suppresses bootstrap. GAE carry stops at either termination flag. A rollout boundary is not an episode ending. One-step TD here means a one-transition target, not TD(lambda) with lambda=1.','',
        f'The configuration audit finds only these changed fields: {", ".join(summary["configuration_differences"])}. Both arms use seed 101, actor 4-64-Tanh-2, critic 4-64-Tanh-1, actor Adam LR .0003, critic Adam LR .001, clip ratio .2, 2048-transition rollouts over 8 lanes, 10 epochs, minibatches of 256 and gradient bound .5. The critic change can affect later GAE values and sampled policies even though the actor formula is unchanged.','',
        '## Training and complete-episode results','',
        '| Critic label | Actual train steps | Complete train episodes | Last 100 reward mean | First 100-window >=475 | First 100-window =500 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in RUNS:
        s=arms[arm]['stats'];c=s['first_full100_window_crossing_steps']
        c475=f'{c["475"]:,}' if c['475'] is not None else 'Not reached'
        c500=f'{c["500"]:,}' if c['500'] is not None else 'Not reached'
        lines.append(f'| {LABELS[arm]} | {s["training_steps"]:,} | {s["complete_training_episodes"]:,} | {s["train_last100_mean"]:.3f} | {c475} | {c500} |')
    lines += ['', '![Full and early training reward comparison](../figures/td1_critic_training.png)','',
        'Figure 1. Raw complete-episode rewards are faint; trailing 100-episode means are solid. Before 100 episodes the mean uses all available episodes. Both panels use actual interactions; the right panel zooms to 400,000 and marks each first full 100-episode mean of 500. Crossing a training-window threshold is descriptive and does not evaluate a frozen intermediate policy. PPO episodes can span multiple updates.','',
        f'After first reaching a full 100-episode mean of 500, the next {a["training_stability_after_first_full500_window"]["subsequent_complete_episodes"]:,} GAE-label episodes and {b["training_stability_after_first_full500_window"]["subsequent_complete_episodes"]:,} one-step-TD-label episodes all score 500: neither arm collapses later in this recorded run. The one-step label reaches that first window {b["first_full100_window_crossing_steps"]["500"]-a["first_full100_window_crossing_steps"]["500"]:,} interactions later. This is a single-seed trajectory comparison, not a population sample-efficiency estimate.','',
        '## Critic training losses','',
        '![Different-label critic fitting losses](../figures/td1_critic_training_mse.png)','',
        'Figure 2. Each panel uses twice the logged half-MSE: the unweighted average of pre-update minibatch MSE observations over all reuse epochs. The solid curve averages the most recent 20 updates, with a growing window initially. This is not a final full-rollout MSE after optimization. The labels and their noise/bias differ, so a smaller training loss does not establish more accurate state values. Zero losses, if any, are masked on the logarithmic axis.',
        f'First-update own-label MSE is {a["training_target_mse_first_update"]:.6g} for GAE versus {b["training_target_mse_first_update"]:.6g} for one-step TD. Last-update values are {a["training_target_mse_last_update"]:.6g} and {b["training_target_mse_last_update"]:.6g}. The much smaller initial one-step fitting loss does not translate into faster learning or smaller final initial-state prediction discrepancy in this run.','',
        '## Fresh frozen-policy evaluation','',
        'Both models are frozen before 500 complete stochastic-action episodes on seeds 88000000-88000499. Original rewards and the official 500-step limit are retained. No rollout slice is counted as an episode.','',
        '| Critic label | Mean raw reward | Sample SD | Range | At 500-step cap |',
        '|---|---:|---:|---|---:|']
    for arm in RUNS:
        s=arms[arm]['stats'];lines.append(f'| {LABELS[arm]} | {s["evaluation_mean"]:.3f} | {s["evaluation_std_sample"]:.3f} | [{s["evaluation_min"]:.0f}, {s["evaluation_max"]:.0f}] | {s["evaluation_cap_hits"]}/500 |')
    lines += ['', '![Complete fresh evaluation rewards](../figures/td1_critic_evaluation.png)','',
        'Figure 3. Identical axes and bins show every full-episode reward. SD uses ddof=1. If both methods reach the score cap, this evaluation cannot distinguish their robustness beyond 500 steps; it does not force a winner from training curves.','',
        '## A common initial-state prediction diagnostic','',
        '| Critic label | States | Mean V(s0) | Mean observed G0 | Bias V-G | RMSE |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in RUNS:
        d=arms[arm]['stats']['initial_value_diagnostic'];lines.append(f'| {LABELS[arm]} | {d["n"]} | {d["mean_prediction"]:.7f} | {d["mean_mc_return"]:.7f} | {d["bias"]:.7g} | {d["rmse"]:.7g} |')
    lines += ['', '![Initial-state prediction errors](../figures/td1_critic_initial_value.png)','',
        'Figure 4. Reset states are identical across arms, but each Monte Carlo return follows its own frozen policy and one sampled trajectory. Predictions use no gradients. These are errors relative to observed returns, not known true state values. One sampled return per state can contain Monte Carlo noise; here all observed G0 values are identical because all episodes reach the cap, so observed error variation comes from predictions across states. These are not the training losses in Figure 2. At a 500-step truncation, the maximum omitted continuation contribution at reset is .95^500/(1-.95), below 1e-9.',
        f'Both critics slightly underpredict the observed initial-state returns. The GAE-label critic has bias {a["initial_value_diagnostic"]["bias"]:.7f} and RMSE {a["initial_value_diagnostic"]["rmse"]:.7f}; one-step TD has bias {b["initial_value_diagnostic"]["bias"]:.7f} and RMSE {b["initial_value_diagnostic"]["rmse"]:.7f}. The latter discrepancy is larger in this held-out set, despite its smaller initial own-label training loss. This does not identify the exact conditional value error or generalize across training seeds.','',
        '## Mechanism validation','',
        f'Independent hand targets match one-step TD with maximum error {validation["targets"]["hand_max_absolute_error"]:.1f}. On the matched first batch, actor parameters remain identical (error {validation["matched_first_update"]["first_batch_actor_parameter_max_absolute_error"]:.1f}), while critic parameters differ by {validation["matched_first_update"]["first_batch_critic_parameter_max_absolute_difference"]:.8f}; targets remain frozen throughout reuse. A separate 143-step interruption/resume check reproduces all numerical state exactly. These tests support the intended label-only intervention.','',
        '## Scope and interpretation','',
        f'Both arms use {a["training_steps"]:,} actual interactions and {a["rollout_updates"]} rollout updates. The baseline uses {a["actor_optimizer_steps"]:,} actor and {a["critic_optimizer_steps"]:,} critic steps; one-step TD uses {b["actor_optimizer_steps"]:,} and {b["critic_optimizer_steps"]:,}. Recorded training times are {a["training_elapsed_seconds"]:.3f} and {b["training_elapsed_seconds"]:.3f} seconds, not repeated controlled timing estimates. Each arm accounts separately for ongoing episodes at the budget.',
        'This is a one-training-seed ablation. Matching architecture, seed and hyperparameters isolates the implemented label change for this run, but does not establish superiority across training seeds. Final reward SD, critic own-target fitting loss and initial-state Monte Carlo prediction RMSE are three different statistics. Final evaluation does not select checkpoints.','',
        '## 中文直观解释','',
        'Actor 的优势估计仍是 GAE；只把 Critic 的“参考答案”从 GAE 推导的多步标签换成一步 TD：先看当前奖励，再加旧 Critic 对下一状态的折扣估计。训练图展示分数如何随真实交互步数变化，前 40 万步放大图标出两种方法首次完整 100 回合窗口均分达到 500 的位置。',
        f'这个 seed 下，GAE 标签首次最近 100 回合平均达到 475/500 分分别在 {a["first_full100_window_crossing_steps"]["475"]:,}/{a["first_full100_window_crossing_steps"]["500"]:,} 步，一步 TD 是 {b["first_full100_window_crossing_steps"]["475"]:,}/{b["first_full100_window_crossing_steps"]["500"]:,} 步，前者学得更早。达到 500 窗口后两者都没有再出现低于 500 的完整训练回合。最后各自 500 个新种子评估均为 500 分、样本 SD 为 0，最终观测成绩相同。',
        '两种参考答案不同，不能因为第二张图的训练 MSE 更低就说 Critic 更准。第四张图改用相同初始状态的实际折扣回报作为共同诊断，但每个状态仍只有各自策略的一条随机轨迹，并非已知真值。最终直方图要单独看；若都达到 500 分，只能说这批完整评估都碰到了上限。','',
        '## Reproduction and evidence','',
        '```sh',
        '.venv/bin/python -m src.compare_td1_critic',
        '```','',
        'The command only reads frozen checkpoints, configurations, complete training/evaluation CSVs and initial-value diagnostics, then rebuilds four figures and this document. It does not train or modify the original report. Raw source paths and all counters are retained in [comparison metrics](../data/td1_critic_comparison.json). Validation evidence: [one-step TD critic validation](../data/validation/td1_critic.json).']
    (root/'docs/TD1_CRITIC_COMPARISON.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);args=p.parse_args();root=args.root.resolve()
    arms={arm:load_arm(root,arm) for arm in RUNS}
    ca=arms['gae']['stats']['configuration'];cb=arms['td1']['stats']['configuration']
    differences={k:{'gae':ca.get(k),'td1':cb.get(k)} for k in sorted(set(ca)|set(cb)) if ca.get(k)!=cb.get(k)}
    check(set(differences)=={'critic_target','value_target'},f'Unexpected non-label configuration differences: {differences}')
    check(np.array_equal(arms['gae']['states'],arms['td1']['states']),'Initial evaluation states are not exactly paired')
    check(arms['gae']['stats']['training_steps']==arms['td1']['stats']['training_steps'],'Actual interaction budgets differ')
    validation=read_json(root/'data/validation/td1_critic.json');check(validation['status']=='passed','One-step TD validation has not passed')
    summary=dict(complete=True,actor_advantage='GAE unchanged, normalized within rollout',configuration_differences=differences,evaluation_seed_base=88000000,evaluation_episodes_per_arm=500,evaluation_action_rule='sample categorical policy',arms={arm:x['stats'] for arm,x in arms.items()},initial_states_exactly_paired=True,mc_prediction_scope='one observed return under each own frozen policy; not known true values',limitations=['One training seed','Different training targets make own-label MSE incomparable as truth accuracy','Full reward cap may hide differences beyond 500 steps'],source_paths={arm:{'run':f'runs/{name}','fresh_evaluation':f'runs/td1_critic_eval/{arm}','initial_values':f'runs/td1_critic_eval/{arm}/initial_value_diagnostic.csv'} for arm,name in RUNS.items()})
    summary['training_window_diagnostic_scope']='One-seed descriptive complete-episode training windows; not frozen-intermediate-policy evaluations or population sample-efficiency estimates.'
    summary['early_training_zoom_steps']=400000
    make_plots(root,arms);(root/'data/td1_critic_comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');write_doc(root,arms,summary)
    print(json.dumps(dict(complete=True,arms={a:dict(mean=x['stats']['evaluation_mean'],sd=x['stats']['evaluation_std_sample']) for a,x in arms.items()},output='data/td1_critic_comparison.json'),indent=2))

if __name__=='__main__':main()
