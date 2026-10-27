"""Prepare explicitly partitioned flight logs for the revised DSFN protocol."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from dsfn.data import CHANNEL_NAMES, ROLES, WINDOW_LENGTH, dataset_summary, load_dataset

DATASETS = {
    'rflymad': dict(dataset_id='RflyMAD', class_names=['Normal','Accel','GPS','Gyro','Mag','Baro'], sampling_hz=None, validation_stride=1024),
    'uavfd': dict(dataset_id='UAV-FD', class_names=['Normal','Damage5','Damage10'], sampling_hz=350., validation_stride=512),
}


def read_manifest(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {'log_id','group_id','split'}.issubset(rows[0]):
        raise ValueError('Manifest needs log_id, group_id and split columns')
    logs, groups = set(), {}
    for row in rows:
        log, group, split = row['log_id'], row['group_id'], row['split']
        if not log or not group or split not in ROLES or log in logs:
            raise ValueError('Log IDs must be unique; every row needs a valid group/split')
        logs.add(log);groups.setdefault(group,set()).add(split)
    if any(len(value) != 1 for value in groups.values()):
        raise ValueError('Assign every flight-case group to one split before windowing')
    if set(r['split'] for r in rows) != set(ROLES):
        raise ValueError('Manifest must contain train, val and test flights')
    return rows


def prepare(flights_dir, manifest, output_dir, dataset, sampling_hz=None):
    settings=DATASETS[dataset];rows=read_manifest(manifest)
    flights_dir=Path(flights_dir).resolve(strict=True);output_dir=Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Output directory is not empty; preserve existing data')
    windows, labels, groups, metadata = [], [], [], []
    ids={role:[] for role in ROLES};skipped_transitions=0
    for row in rows:
        log,group,split=row['log_id'],row['group_id'],row['split']
        path=(flights_dir/(log+'.npz')).resolve(strict=True)
        try:path.relative_to(flights_dir)
        except ValueError as exc:raise ValueError('Manifest path escapes --flights-dir') from exc
        with np.load(path,allow_pickle=False) as flight:
            signal=np.asarray(flight['signals'],dtype=np.float32)
            if signal.ndim!=2 or signal.shape[1]!=9 or not len(signal) or not np.isfinite(signal).all():
                raise ValueError('%s signals must be finite [time,9]' % log)
            for key,expected in [('log_id',log),('group_id',group)]:
                if key in flight and str(flight[key].item())!=expected:
                    raise ValueError('%s disagrees with the explicit manifest' % key)
            if 'labels' in flight:
                target=np.asarray(flight['labels'])
                if target.ndim!=1 or len(target)!=len(signal) or target.dtype.kind not in 'iu':
                    raise ValueError('%s labels must be an aligned integer vector' % log)
            elif 'label' in flight:
                label=flight['label']
                if label.ndim!=0 or label.dtype.kind not in 'iu':raise ValueError('label must be an integer scalar')
                target=np.full(len(signal),int(label.item()),dtype=np.int64)
            else:raise ValueError('%s needs label or labels' % log)
            if target.min()<0 or target.max()>=len(settings['class_names']):raise ValueError('Label outside class map')
            stride=512 if split=='train' else settings['validation_stride'] if split=='val' else 1024
            found=0
            for start in range(0,len(signal)-WINDOW_LENGTH+1,stride):
                stop=start+WINDOW_LENGTH;part=target[start:stop]
                if not np.all(part==part[0]):skipped_transitions+=1;continue
                index=len(windows);windows.append(signal[start:stop].T.copy())
                labels.append(int(part[0]));groups.append(group);ids[split].append(index)
                metadata.append(dict(window_id=index,log_id=log,group_id=group,split=split,start_sample=start,stop_sample=stop))
                found+=1
            if not found:raise ValueError('%s has no complete homogeneous 1,024-sample window' % log)
    sampling=settings['sampling_hz'] if sampling_hz is None else float(sampling_hz)
    if sampling is not None and (not np.isfinite(sampling) or sampling<=0):raise ValueError('Sampling frequency must be positive')
    if dataset=='uavfd' and sampling!=350.:raise ValueError('The registered UAV-FD comparison uses synchronized 350 Hz signals')
    spec=dict(schema='dsfn_grouped_flight_windows_v1',dataset_id=settings['dataset_id'],num_classes=len(settings['class_names']),
        class_names=settings['class_names'],input_shape=[9,WINDOW_LENGTH],sampling_hz=sampling,channel_order=list(CHANNEL_NAMES),
        window_length=WINDOW_LENGTH,train_stride=512,validation_stride=settings['validation_stride'],test_stride=1024,
        split_assignment='explicit flight_manifest.csv before window extraction',
        label_policy='complete homogeneous windows; transition windows excluded',skipped_transition_windows=skipped_transitions)
    output_dir.mkdir(parents=True,exist_ok=True)
    np.save(output_dir/'windows.npy',np.stack(windows).astype(np.float32));np.save(output_dir/'labels.npy',np.asarray(labels,dtype=np.int64))
    np.save(output_dir/'groups.npy',np.asarray(groups,dtype=str))
    (output_dir/'dataset.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
    (output_dir/'splits.json').write_text(json.dumps(ids,indent=2)+'\n',encoding='utf-8')
    with (output_dir/'window_metadata.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(metadata[0]));writer.writeheader();writer.writerows(metadata)
    with (output_dir/'flight_manifest.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['log_id','group_id','split']);writer.writeheader()
        writer.writerows({k:row[k] for k in writer.fieldnames} for row in rows)
    return dataset_summary(load_dataset(output_dir))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=sorted(DATASETS),default='rflymad')
    parser.add_argument('--flights-dir',type=Path);parser.add_argument('--manifest',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--sampling-hz',type=float)
    parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    if args.verify_only:result=dataset_summary(load_dataset(args.output_dir))
    else:
        if args.flights_dir is None or args.manifest is None:parser.error('--flights-dir and --manifest are required unless --verify-only is used')
        result=prepare(args.flights_dir,args.manifest,args.output_dir,args.dataset,args.sampling_hz)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
