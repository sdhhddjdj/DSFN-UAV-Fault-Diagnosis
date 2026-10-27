"""Prepare the registered UAV-FD raw-MAT comparison without training.

Ported from the actual 2026-10-26 preparation. Default operation is offline.
Test windows use every other valid 512-step window within each contiguous
flight/segment chain, restarting after gaps, matching the final 68-window test.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import urlopen

import numpy as np

from dsfn.data import load_dataset, dataset_summary

RECORD = 'https://zenodo.org/api/records/7648996'
FS = 350.0
LENGTH = 1024
STRIDE = 512
CLASSES = ['Normal', 'Damage5', 'Damage10']
CHANNELS = ['Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Accel_X', 'Accel_Y', 'Accel_Z', 'Vel_N', 'Vel_E', 'Vel_D']
EXPECTED_COUNTS = {'train': 555, 'val': 123, 'test': 68}

def labels(d, name):
    return [str(a.item()) for a in np.asarray(d[name + '_label']).ravel()]

def stream(d, key, fields, audit):
    topic = re.sub(r'_\d+$', '', key)
    names = labels(d, topic)
    ar = np.asarray(d[key], dtype=np.float64)
    ti = names.index('TimeUS')
    vals = ar[:, [names.index(x) for x in fields]]
    t = ar[:, ti] / 1e6
    finite = np.isfinite(t) & np.isfinite(vals).all(axis=1)
    if not finite.all(): raise ValueError(key + ' nonfinite input rows')
    if np.any(np.diff(t) < 0): raise ValueError(key + ' reversed timestamps')
    unique, indices = np.unique(t, return_index=True)
    vals = vals[indices]
    dt = np.diff(unique)
    audit[key] = {'rows': len(ar), 'unique_times':len(unique), 'duplicate_times':len(ar)-len(unique),
                  'start_s':float(t[0]), 'end_s':float(t[-1]),
                  'average_hz':float((len(unique)-1)/(unique[-1]-unique[0])),
                  'median_hz':float(1/np.median(dt)), 'max_gap_s':float(dt.max()),
                  'fields':fields}
    return unique, vals

def zoh(t, v, grid, max_age):
    idx = np.searchsorted(t, grid, side='right') - 1
    safe = np.clip(idx, 0, len(t)-1)
    valid = (idx>=0) & (grid<=t[-1]) & ((grid-t[safe])<=max_age)
    return v[safe], valid

def airborne_intervals(d, audit):
    ev = np.asarray(d['EV'])
    nm = labels(d, 'EV')
    events = [(float(r[nm.index('TimeUS')])/1e6, int(r[nm.index('Id')])) for r in ev]
    audit['events'] = events
    intervals=[]; start=None
    for t, code in sorted(events):
        if code == 28:
            if start is None: start=t
        elif code in (18, 11):
            if start is not None:
                if t-start>4: intervals.append((start+2, t-2))
                start=None
    if start is not None:
        raise ValueError('Unclosed airborne interval: no LAND_COMPLETE/DISARMED after NOT_LANDED')
    if not intervals: raise ValueError('No explicit EV28-to-EV18/11 airborne interval')
    audit['airborne_intervals_trimmed_s'] = intervals
    audit['airborne_rule'] = 'EV28 NOT_LANDED to first EV18 LAND_COMPLETE or EV11 DISARMED; trim 2s at each edge'
    return intervals

def gps_absolute(d, audit):
    nm=labels(d,'GPS'); g=np.asarray(d['GPS_0'],dtype=float)
    status=g[:,nm.index('Status')]; good=(status>=3)&(g[:,nm.index('GWk')]>0)
    if good.sum()<2: raise ValueError('Insufficient GPS absolute time')
    boot=g[good,nm.index('TimeUS')]/1e6
    absolute=g[good,nm.index('GWk')]*604800+g[good,nm.index('GMS')]/1000
    off=absolute-boot
    if np.ptp(off)>.5: raise ValueError('GPS/boot-time offset instability over 0.5s')
    audit['gps']={'week':int(g[good,nm.index('GWk')][0]),'start_gps_seconds':float(absolute[0]),
                  'end_gps_seconds':float(absolute[-1]), 'boot_to_gps_offset_s':float(np.median(off)),
                  'offset_spread_s':float(np.ptp(off))}
    return float(np.median(off))

def identity(name):
    m=re.fullmatch(r'NO_FAULT([1-6])\.mat', name)
    if m: return int(m[1]),0,0
    m=re.fullmatch(r'FAULT_M([1-6])_(5|10)\.mat', name)
    if not m: raise ValueError(name)
    return int(m[1]), int(m[2]), 1 if m[2]=='5' else 2

def prepare_one(path):
    from scipy.io import loadmat
    required=['IMU_0','IMU_1','IMU_2','IMU_label','XKF1_0','XKF1_1','XKF1_label',
              'RCOU','RCOU_label','CTUN','CTUN_label','EV','EV_label','GPS_0','GPS_label','label_fault']
    d=loadmat(path, variable_names=required)
    group,severity,lab=identity(path.name)
    audit={'file':path.name,'group_number':group,'severity_code':severity,'class_id':lab,
           'split':'train' if group<=4 else ('val' if group==5 else 'eval'), 'topics':{}}
    original=np.unique(d['label_fault']).astype(int).tolist() if 'label_fault' in d else None
    audit['original_label_fault_values']=original
    audit['label_source']='Official main.m filelist + faultCode [0*6,5*6,10*6], remapped 0/1/2; optional embedded label_fault cross-check'
    if original is not None and original != [lab]: raise ValueError(path.name + ' unexpected label_fault ' + str(original))
    streams=[]
    for i in range(3):
        streams.append(stream(d,'IMU_'+str(i),['GyrX','GyrY','GyrZ','AccX','AccY','AccZ'],audit['topics']))
    for i in range(2):
        streams.append(stream(d,'XKF1_'+str(i),['VN','VE','VD'],audit['topics']))
    rt,rv=stream(d,'RCOU',['C9','C10','C11','C12','C13','C14'],audit['topics'])
    ct,cv=stream(d,'CTUN',['Alt','ThO'],audit['topics'])
    intervals=airborne_intervals(d,audit)
    gps_offset=gps_absolute(d,audit)
    audit['absolute_airborne_intervals_gps_s']=[[a+gps_offset,b+gps_offset] for a,b in intervals]
    windows=[]; meta=[]; n_reject=0; n_candidates=0; pwm_stats=[]
    common_start=max(t[0] for t,v in streams)
    common_end=min(t[-1] for t,v in streams)
    for segment,(a,b) in enumerate(intervals):
        a=max(a,common_start,rt[0],ct[0]); b=min(b,common_end,rt[-1],ct[-1])
        if b-a<LENGTH/FS: continue
        grid=a+np.arange(int(np.floor((b-a)*FS))+1)/FS
        aligned=[]; valid=np.ones(len(grid),dtype=bool)
        for i,(t,v) in enumerate(streams):
            values,ok=zoh(t,v,grid,.05 if i<3 else .1)
            aligned.append(values); valid &=ok
        # Use authors' redundant-IMU / dual-EKF averages AFTER actual-time synchronization.
        sensors=np.mean(np.stack(aligned[:3]),axis=0)
        velocity=np.mean(np.stack(aligned[3:]),axis=0)
        x=np.concatenate([sensors,velocity],axis=1).astype(np.float32)
        pwm,pok=zoh(rt,rv,grid,.1); alt,aok=zoh(ct,cv,grid,.5)
        # Author's PWM criterion, intersected with explicit flight-state events.
        pmean=pwm.mean(axis=1); active=pmean>=1450
        valid &= pok & aok & active
        pwm_stats.append({'segment':segment,'pwm_min':float(pmean.min()),'pwm_median':float(np.median(pmean)),
                          'fraction_pwm_ge1450':float(active.mean()),'alt_min_m':float(alt[:,0].min()),
                          'alt_max_m':float(alt[:,0].max())})
        for start in range(0,len(grid)-LENGTH+1,STRIDE):
            n_candidates+=1
            if not valid[start:start+LENGTH].all(): n_reject+=1; continue
            windows.append(x[start:start+LENGTH].T.copy())
            meta.append({'file':path.name,'flight_id':path.stem,'group_number':group,
                         'motor':0 if severity==0 else group,'severity_code':severity,'class_id':lab,
                         'class_name':CLASSES[lab],'split':audit['split'],'segment':segment,
                         'segment_start_sample':start, 'segment_stop_sample':start+LENGTH,
                         'start_boot_s':float(grid[start]),'end_boot_s':float(grid[start+LENGTH-1]),
                         'start_gps_s':float(grid[start]+gps_offset),'end_gps_s':float(grid[start+LENGTH-1]+gps_offset)})
    audit['flight_context']=pwm_stats
    audit['candidate_windows']=n_candidates; audit['rejected_windows']=n_reject; audit['retained_windows']=len(windows)
    if not windows: raise ValueError(path.name+' has no eligible complete flight windows')
    return windows,meta,audit


def select_nonoverlapping_test(rows):
    """Original final-test selection: restart 0,2,4,... at every valid-chain gap.

    Choosing start_sample modulo 1,024 would be different when the first valid
    window starts at 512 or after a rejected timestamp/PWM region.
    """
    grouped = {}
    for row in rows:
        if row['split'] in ('eval', 'test'):
            grouped.setdefault((row['flight_id'], row['segment']), []).append(row)
    chosen, chains = [], []
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda row: float(row['start_boot_s']))
        chain = []
        for row in values:
            if chain:
                gap = (float(row['start_boot_s']) - float(chain[-1]['start_boot_s'])) * FS
                if abs(gap - STRIDE) > .001:
                    chosen.extend(int(r['window_id']) for r in chain[::2])
                    chains.append(dict(flight=key[0], segment=key[1],
                                       original=len(chain), selected=len(chain[::2])))
                    chain = []
            chain.append(row)
        if chain:
            chosen.extend(int(r['window_id']) for r in chain[::2])
            chains.append(dict(flight=key[0], segment=key[1],
                               original=len(chain), selected=len(chain[::2])))
    keep = set(chosen)
    ids = [int(row['window_id']) for row in rows
           if row['split'] in ('eval', 'test') and int(row['window_id']) in keep]
    for flight in sorted({r['flight_id'] for r in rows if int(r['window_id']) in keep}):
        values = sorted((r for r in rows if r['flight_id'] == flight and int(r['window_id']) in keep),
                        key=lambda row: float(row['start_boot_s']))
        if any(float(b['start_boot_s']) - float(a['end_boot_s']) < 1/FS - 1e-7
               for a, b in zip(values, values[1:])):
            raise ValueError('Selected test intervals overlap in ' + flight)
    return ids, chains


def noise_replay_metadata(rows, selected_test):
    """Index the selected test windows within the original stride-512 test bank."""
    original_ids = [int(row['window_id']) for row in rows if row['split'] in ('eval', 'test')]
    keep = set(selected_test)
    positions = [position for position, window_id in enumerate(original_ids) if window_id in keep]
    if len(positions) != len(keep):
        raise ValueError('A selected test window is absent from the original test bank')
    return dict(test_noise_original_count=len(original_ids), test_noise_positions=positions,
                test_noise_seed=20261021)


def _expected_files():
    return (['NO_FAULT%s.mat' % i for i in range(1, 7)] +
            ['FAULT_M%s_%s.mat' % (i, severity) for i in range(1, 7) for severity in (5, 10)])


def _download_file(entry, raw_dir):
    path = raw_dir / entry['key']
    expected = entry['checksum'].split(':')[-1]
    def verified(candidate):
        if not candidate.is_file() or candidate.stat().st_size != entry['size']:
            return False
        value = hashlib.md5()
        with candidate.open('rb') as handle:
            for block in iter(lambda: handle.read(1024*1024), b''):
                value.update(block)
        return value.hexdigest() == expected
    if verified(path):
        return
    part = path.with_suffix('.mat.part')
    for attempt in range(4):
        try:
            with urlopen(entry['links']['self'], timeout=90) as response, part.open('wb') as handle:
                for block in iter(lambda: response.read(1024*1024), b''):
                    handle.write(block)
            if not verified(part):
                raise ValueError('Official checksum/size mismatch: ' + path.name)
            part.replace(path)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2)


def download_raw(raw_dir):
    """Explicit network operation, called only with --download."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(RECORD, timeout=60) as response:
        record = json.load(response)
    expected = set(_expected_files())
    entries = [entry for entry in record['files'] if entry['key'] in expected]
    if {entry['key'] for entry in entries} != expected:
        raise ValueError('The Zenodo record does not contain the expected 18 MAT files')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda entry: _download_file(entry, raw_dir), entries))
    return record


def _absolute_overlap_audit(audits):
    airborne, raw_logs = [], []
    for i, a in enumerate(audits):
        for b in audits[i+1:]:
            duration = min(a['gps']['end_gps_seconds'], b['gps']['end_gps_seconds']) - max(
                a['gps']['start_gps_seconds'], b['gps']['start_gps_seconds'])
            if duration > .5:
                raw_logs.append(dict(a=a['file'], b=b['file'], overlap_s=duration))
            for aa, ab in a['absolute_airborne_intervals_gps_s']:
                for ba, bb in b['absolute_airborne_intervals_gps_s']:
                    duration = min(ab, bb) - max(aa, ba)
                    if duration > .5:
                        airborne.append(dict(a=a['file'], b=b['file'], overlap_s=duration))
    return airborne, raw_logs


def prepare(raw_dir, output_dir):
    raw_dir = Path(raw_dir).resolve(strict=True)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Preserve existing data; output directory must be empty')
    paths = sorted(raw_dir / name for name in _expected_files())
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing official files: ' + ', '.join(missing))
    windows, rows, audits = [], [], []
    for path in paths:
        flight_windows, metadata, audit = prepare_one(path)
        for values, row in zip(flight_windows, metadata):
            row['window_id'] = len(rows)
            windows.append(values); rows.append(row)
        audits.append(audit)
        print('Prepared', path.name, len(flight_windows), 'stride-512 candidate windows', flush=True)
    airborne, raw_logs = _absolute_overlap_audit(audits)
    if airborne or raw_logs:
        raise ValueError('Absolute flight/log overlap requires review; no arrays written')
    selected_test, chains = select_nonoverlapping_test(rows)
    selected_test = set(selected_test)
    replay = noise_replay_metadata(rows, selected_test)
    keep = [i for i, row in enumerate(rows) if row['split'] != 'eval' or i in selected_test]
    final_rows, ids = [], {role: [] for role in ('train', 'val', 'test')}
    for new_id, old_id in enumerate(keep):
        row = dict(rows[old_id]);role = 'test' if row['split'] == 'eval' else row['split']
        row.update(window_id=new_id, original_window_id=old_id, split=role,
                   log_id=row['flight_id'], group_id=row['flight_id'],
                   # Boot-time indices keep separate airborne segments separate.
                   start_sample=int(round(float(row['start_boot_s']) * FS)))
        row['stop_sample'] = row['start_sample'] + LENGTH
        final_rows.append(row);ids[role].append(new_id)
    counts = {role: len(values) for role, values in ids.items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError('Prepared counts %s differ from registered counts %s; review source conversion' % (counts, EXPECTED_COUNTS))
    x = np.stack([windows[i] for i in keep]).astype(np.float32)
    y = np.asarray([r['class_id'] for r in final_rows], dtype=np.int64)
    groups = np.asarray([r['flight_id'] for r in final_rows], dtype=str)
    spec = dict(schema='uavfd_registered_raw_mat_v1', dataset_id='UAV-FD', num_classes=3,
                class_names=CLASSES, input_shape=[9,LENGTH], sampling_hz=FS, channel_order=CHANNELS,
                window_length=LENGTH, train_stride=512, validation_stride=512, test_stride=1024,
                source=RECORD, split_counts=counts,
                split_rule='Index/motor positions 1-4 train, 5 validation, 6 test; all severities follow the same assignment',
                synchronization='Per-topic TimeUS to 350 Hz zero-order hold before redundant-sensor averages',
                inertial_source='Mean of synchronized IMU_0/IMU_1/IMU_2',
                velocity_source='Mean of synchronized XKF1_0/XKF1_1 VN VE VD (already NED)',
                flight_rule='EV28 to first EV18/11, 2-second edge trim; PWM mean >=1450; complete valid windows only',
                test_selection='Within each flight/segment and contiguous valid stride512 chain, retain chronological positions 0,2,4,...; restart after gaps',
                normalization='Raw physical units; trainer fits channel statistics on train windows only',
                **replay)
    audit = dict(files=audits, original_test_windows=sum(r['split']=='eval' for r in rows),
                 final_test_windows=len(selected_test), contiguous_test_chains=chains,
                 selected_original_test_window_ids=sorted(selected_test), counts=counts,
                 overlapping_absolute_flight_intervals=airborne, overlapping_full_log_gps_intervals=raw_logs)
    output_dir.mkdir(parents=True,exist_ok=True)
    np.save(output_dir/'windows.npy',x);np.save(output_dir/'labels.npy',y);np.save(output_dir/'groups.npy',groups)
    for name,value in [('dataset.json',spec),('splits.json',ids),('data_audit.json',audit)]:
        (output_dir/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    with (output_dir/'window_metadata.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(final_rows[0]));writer.writeheader();writer.writerows(final_rows)
    flight_rows=[dict(log_id=a['file'][:-4],group_id=a['file'][:-4],
                      split='test' if a['split']=='eval' else a['split']) for a in audits]
    with (output_dir/'flight_manifest.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['log_id','group_id','split']);writer.writeheader();writer.writerows(flight_rows)
    return dataset_summary(load_dataset(output_dir))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,required=True,help='Directory containing the official 18 MAT flight logs')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--download',action='store_true',help='Explicitly download/checksum official Zenodo files first')
    args=parser.parse_args()
    record=download_raw(args.raw_dir) if args.download else None
    summary=prepare(args.raw_dir,args.output_dir)
    if record is not None:
        (args.output_dir/'zenodo_record.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
