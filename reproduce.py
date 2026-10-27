"""Train/evaluate the revised DSFN and eight comparison models."""
import argparse
import json
from pathlib import Path
from dsfn.protocol import FAMILIES, SEEDS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('train','evaluate','summarize'))
    p.add_argument('--dataset',choices=('rflymad','uavfd'),required=True)
    p.add_argument('--data',type=Path)
    p.add_argument('--config',type=Path)
    p.add_argument('--family',choices=('all',)+FAMILIES,default='dsfn')
    p.add_argument('--seeds',default=','.join(map(str,SEEDS)))
    p.add_argument('--output',type=Path,default=Path('runs'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--split',choices=('val','test'),default='test')
    p.add_argument('--noise-bank',type=Path,help='Optional archived Gaussian bank for the selected split')
    p.add_argument('--allow-custom-split',action='store_true')
    a=p.parse_args(); seeds=[int(s) for s in a.seeds.split(',')]
    families=list(FAMILIES) if a.family=='all' else [a.family]
    if a.command=='summarize':
        from dsfn.evaluate import summarize_runs
        summarize_runs(a.output,a.dataset,families,seeds,a.split)
        return
    if a.data is None:p.error('--data is required for train/evaluate')
    config=json.loads((a.config or Path(__file__).parent/'configs'/(a.dataset+'.json')).read_text(encoding='utf-8'))
    if config['dataset']!=a.dataset:p.error('Config dataset mismatch')
    from dsfn.data import load_dataset
    from dsfn.training import validate_config,train_one,evaluate_one
    ctx=load_dataset(a.data);validate_config(ctx,config,a.allow_custom_split)
    for family in families:
        for seed in seeds:
            if a.command=='train':train_one(ctx,config,family,seed,a.output,a.device,a.noise_bank)
            else:evaluate_one(ctx,config,family,seed,a.output,a.device,a.split,a.noise_bank)

if __name__=='__main__':main()
