"""Portable training entry points; importing this module never starts a run."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from .augmentation import loader, ModalityNoise
from .protocol import set_seed, EvaluationBank, CONDITIONS, metrics
from .evaluate import predict, save_json, save_csv, export_evaluation
from . import models


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def bank_for(ctx, config, split, noise_path=None):
    defaults = {'rflymad': {'val': 20261015, 'test': 20261520},
                'uavfd': {'val': 20261020, 'test': 20261021}}
    seed = config.get(split+'_noise_seed', ctx['spec'].get(split+'_noise_seed',defaults[config['dataset']][split]))
    return EvaluationBank(ctx, split, seed, noise_path)


def validate_config(ctx, config, allow_custom_split=False):
    if int(ctx['spec']['num_classes']) != config['num_classes'] or list(ctx['spec']['class_names']) != config['class_names']:
        raise ValueError('Dataset classes/order differ from the selected config')
    e = config['expected_counts']; wanted = {'train':e['train'],'val':e['validation'],'test':e['test']}
    actual = {k:len(v) for k,v in ctx['ids'].items()}
    if actual != wanted and not allow_custom_split:
        raise ValueError('Window counts differ from the manuscript: %s != %s. Use --allow-custom-split for a different experiment.' % (actual,wanted))
    if tuple(config['gtd_scales']) != (0.,.25,1.):
        raise ValueError('This release uses GTD inference scales (0,.25,1)')


def train_one(ctx, config, family, seed, output, device='cpu', val_noise_bank=None):
    out=Path(output)/config['dataset']/family/('seed_%s'%seed)
    if out.exists():
        raise FileExistsError('Preserve the existing run; choose a new --output directory: '+str(out))
    out.mkdir(parents=True)
    cfg=dict(config,family=family,seed=seed,actual_counts={k:len(v) for k,v in ctx['ids'].items()},
             split_indices={k:v.tolist() for k,v in ctx['ids'].items()},
             split_groups={k:sorted(set(ctx['groups'][v].tolist())) for k,v in ctx['ids'].items()},
             class_weights=ctx['weights'].tolist(),normalization_mean=ctx['mean'].tolist(),
             normalization_scale=ctx['scale'].tolist(),test_used_for_selection=False,
             selection='highest mean validation noise18 macro-F1; earliest tie',
             implementation='portable revised manuscript export')
    save_json(out/'config.json',cfg)
    np.savez(out/'normalizer.npz',mean=ctx['mean'],scale=ctx['scale'])
    val=bank_for(ctx,config,'val',val_noise_bank)
    if family=='centaur':
        from .centaur_training import train_centaur
        details=train_centaur(ctx,config,seed,out,device,val)
        save_json(out/'training_complete.json',dict(completed=True,**details));return out
    model=models.build_base_model(family,config['num_classes'],seed).to(device)
    torch.save(cpu_state(model),out/'initial_model.pth');set_seed(seed)
    train_loader=loader(ctx,seed,weak=True,batch_size=config['batch_size'])
    augmentation=ModalityNoise(seed);weights=torch.as_tensor(ctx['weights'],device=device)
    ema=models.create_ema(model)
    epochs=config['base_epochs'] if family=='dsfn' else config['baseline_epochs']
    opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=config['min_learning_rate'])
    history=[];best=-1.;best_epoch=0;started=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for x,y in train_loader:
            x,y=augmentation(x).to(device),y.to(device);opt.zero_grad(set_to_none=True)
            loss=models.base_loss(family,model(x),y,weights,epoch=epoch,
                auxiliary_weight=config['lambda_aux'],balance_weight=config['lambda_balance'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();opt.step();models.update_base_ema(ema,model,config['ema_decay'])
            losses.append(float(loss.detach()))
        ls,_=predict(ema,val,device,conditions=CONDITIONS[:19])
        scores=[metrics(val.y,a,config['num_classes'])[0] for a in ls]
        score=float(np.mean([s['macro_f1'] for s in scores[1:]]))
        if score>best:
            best,best_epoch=score,epoch;torch.save(cpu_state(ema),out/'best_model.pth')
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),clean_accuracy=scores[0]['accuracy'],
                 clean_macro_f1=scores[0]['macro_f1'],noise_macro_f1=score,
                 learning_rate=opt.param_groups[0]['lr'],elapsed_seconds=time.monotonic()-started)
        history.append(row);save_csv(out/'history.csv',history);schedule.step()
        print(config['dataset'],family,seed,json.dumps(row),flush=True)
    torch.save(cpu_state(ema),out/('fixed%s.pth'%epochs))
    if family=='dsfn':
        model.load_state_dict(torch.load(out/'best_model.pth',map_location=device,weights_only=True))
        model.zero_grad(set_to_none=True);train_gtd(ctx,config,model,seed,out,device)
    save_json(out/'training_complete.json',dict(completed=True,family=family,seed=seed,
        base_epochs=epochs,gtd_epochs=config['gtd_epochs'] if family=='dsfn' else 0,
        best_epoch=best_epoch,best_validation_noise_macro_f1=best,test_used_for_selection=False))
    return out


def train_gtd(ctx,config,base,seed,out,device):
    model=models.wrap_gtd(base,seed,class_weights=ctx['weights']).to(device);frozen=cpu_state(model.frozen)
    set_seed(seed);train_loader=loader(ctx,seed,weak=False,batch_size=config['batch_size'])
    augmentation=ModalityNoise(seed);weights=torch.as_tensor(ctx['weights'],device=device)
    opt=torch.optim.AdamW(model.adapters.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,config['gtd_epochs'],eta_min=config['min_learning_rate'])
    ema=models.create_ema(model.adapters);history=[];torch.save(cpu_state(model.adapters),out/'gtd_initial.pth')
    for epoch in range(1,config['gtd_epochs']+1):
        model.train();losses=[]
        for clean,y in train_loader:
            noisy,apply,mod,_=augmentation(clean,True);opt.zero_grad(set_to_none=True)
            loss,_=models.gtd_loss(model,clean.to(device),noisy.to(device),y.to(device),weights,
                                   (apply.to(device),mod.to(device)))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite GTD loss')
            loss.backward()
            if any(p.grad is not None for p in model.frozen.parameters()):raise AssertionError('Frozen backbone received gradients')
            opt.step();models.update_gtd_ema(ema,model.adapters,config['ema_decay']);losses.append(float(loss.detach()))
        if any(not torch.equal(v.cpu(),frozen[k]) for k,v in model.frozen.state_dict().items()):raise AssertionError('Frozen backbone/BN changed')
        history.append(dict(epoch=epoch,loss=float(np.mean(losses))));save_csv(out/'gtd_history.csv',history);schedule.step()
        print('GTD',seed,epoch,history[-1],flush=True)
    torch.save(cpu_state(ema),out/('gtd_fixed%s.pth'%config['gtd_epochs']))


def load_run(ctx,config,family,seed,output,device):
    out=Path(output)/config['dataset']/family/('seed_%s'%seed)
    done=json.loads((out/'training_complete.json').read_text(encoding='utf-8'))
    if not done['completed']:raise ValueError('Training is incomplete')
    stored=json.loads((out/'config.json').read_text(encoding='utf-8'))
    for key in ('num_classes','class_names','dataset'):
        if stored[key]!=config[key]:raise ValueError('Run/config mismatch: '+key)
    if stored['split_indices']!={k:v.tolist() for k,v in ctx['ids'].items()} or stored['split_groups']!={k:sorted(set(ctx['groups'][v].tolist())) for k,v in ctx['ids'].items()}:
        raise ValueError('Data split differs from the trained run')
    if not np.array_equal(np.asarray(stored['normalization_mean'],np.float32),ctx['mean']) or not np.array_equal(np.asarray(stored['normalization_scale'],np.float32),ctx['scale']):
        raise ValueError('Training normalization differs from the selected data')
    if family=='centaur':
        from .centaur_training import load_centaur
        return load_centaur(ctx,config,seed,out,device)
    model=models.build_base_model(family,config['num_classes'],seed).to(device)
    model.load_state_dict(torch.load(out/'best_model.pth',map_location=device,weights_only=True),strict=True)
    if family=='dsfn':
        model=models.wrap_gtd(model,seed,ctx['weights']).to(device)
        model.adapters.load_state_dict(torch.load(out/('gtd_fixed%s.pth'%config['gtd_epochs']),map_location=device,weights_only=True),strict=True)
    return model.eval()


def evaluate_one(ctx,config,family,seed,output,device,split='test',noise_bank=None):
    model=load_run(ctx,config,family,seed,output,device);bank=bank_for(ctx,config,split,noise_bank)
    dest=Path(output)/config['dataset']/family/('seed_%s'%seed)/split
    result=export_evaluation(model,bank,config['num_classes'],device,dest,batch_size=16 if family=='centaur' else config['batch_size'])
    save_json(dest/'evaluation.json',dict(split=split,samples=len(bank.y),noise_seed=bank.noise_seed,
                                        supplied_noise_bank=bool(noise_bank),replayed_original_test_subset=bank.replayed_subset,
                                        test_used_for_selection=False))
    return result
