"""Run only unfinished experiments; all final evaluations follow frozen models.

Uses benchmark device choice. Re-run the same command after interruption.
Each training/evaluation process resumes from the last committed batch.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--plan',default='data/experiment_plan.json')
    args=p.parse_args(); plan=json.loads((ROOT/args.plan).read_text())
    decision=json.loads((ROOT/'data/benchmark/decision.json').read_text())
    device=decision['selected']
    (ROOT/'logs/experiment_runner.json').write_text(json.dumps(dict(pid=os.getpid(),started=time.time(),plan=args.plan,device=device),indent=2))
    for run in plan['runs']:
        folder=ROOT/'runs'/run['name']; chosen='cpu' if run['game']=='cartpole' else device
        common=[sys.executable,'-u','src/rl.py']
        cmd=common+['train','--run',str(folder.relative_to(ROOT)),'--game',run['game'],
                    '--seed',str(run['seed']),'--episodes',str(run['episodes']),
                    '--lr',str(run['lr']),'--batch',str(run['batch_size']),'--device',chosen]
        if run['baseline']: cmd+=['--baseline']
        if not (folder/'FROZEN.json').exists():
            print('TRAIN',run['name'],flush=True)
            with (ROOT/'logs'/f"{run['name']}_train.log").open('a') as f:
                subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        summary=folder/'evaluation_summary.json'
        if not summary.exists() or json.loads(summary.read_text()).get('n')!=500:
            print('EVALUATE',run['name'],flush=True)
            with (ROOT/'logs'/f"{run['name']}_evaluate.log").open('a') as f:
                subprocess.run(common+['evaluate','--run',str(folder.relative_to(ROOT)),
                    '--episodes','500','--batch','8','--device',chosen],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('All four formal training and evaluation runs completed.',flush=True)

if __name__=='__main__': main()
