"""Record entropy/saturation on real diagnostic observations of a pilot policy."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.rl import Policy,collect,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    folder=ROOT/args.run; state=torch.load(folder/'final.pt',map_location='cpu',weights_only=False); cfg=state['config']
    model=Policy(cfg['game'],cfg['hidden']);model.load_state_dict(state['model']);model.eval()
    tr=collect(model,cfg['game'],'cpu',list(range(8)),91000000)
    entropy=[];saturated=[]; norms=[]
    with torch.inference_mode():
        for x in tr:
            for lo in range(0,len(x['actions']),512):
                probs=model(torch.as_tensor(np.asarray(x['observations'][lo:lo+512]),dtype=torch.float32)).softmax(-1)
                entropy.extend((-(probs*probs.clamp_min(1e-12).log()).sum(-1)).numpy().tolist())
                saturated.extend((probs.max(-1).values>.999).numpy().tolist())
    result=dict(run=args.run,diagnostic_seed_base=91000000,episodes=8,
                mean_action_entropy_nats=float(np.mean(entropy)),saturated_fraction=float(np.mean(saturated)),
                reward_mean=float(np.mean([sum(x['rewards']) for x in tr])),
                episode_lengths=[len(x['rewards']) for x in tr],
                note='Pilot/tuning diagnostics; excluded from final held-out evaluation. Same diagnostic seeds for pilot comparison.')
    atomic_json(folder/'pilot_diagnostic.json',result); print(json.dumps(result,indent=2))

if __name__=='__main__':main()
