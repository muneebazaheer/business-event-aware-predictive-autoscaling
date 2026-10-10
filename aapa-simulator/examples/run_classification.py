#!/usr/bin/env python3
"""Repaired AAPA Phase 1: resumable, chronological classification reproduction.
Run from repository root. See PHASE1_README.md for details.
"""
import argparse
import json
import logging
import os
import sys
import pickle
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data.features import TimeSeriesFeatureExtractor
from src.models.labeling import WindowedWeakSupervisionLabeler
from src.models.classifier import WorkloadClassifierTrainer

log = logging.getLogger('aapa.phase1')
META = ['function_hash','window_idx','window_start_minute','window_end_minute','day','hour_of_day']

def args_parser():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='data/raw')
    p.add_argument('--output-dir',default='models')
    p.add_argument('--checkpoint-dir',default='data/processed/phase1_checkpoints')
    p.add_argument('--max-functions',type=int,default=0,help='0=all; use 20-50 for smoke test')
    p.add_argument('--min-invocations',type=int,default=1000)
    p.add_argument('--min-active-ratio',type=float,default=.1)
    p.add_argument('--window-size',type=int,default=60)
    p.add_argument('--stride',type=int,default=10)
    p.add_argument('--resume',action='store_true',help='reuse completed per-function checkpoints')
    p.add_argument('--features-only',action='store_true',help='extract/cache features without model training')
    p.add_argument('--smoke-test',action='store_true',help='process 30 eligible functions; uses distinct checkpoint directory')
    return p.parse_args()

def build_index(root):
    # Only HashFunction and Trigger are needed for eligibility. Preserve chronological
    # positions: a missing function-day is 1440 zeros, not a shifted later day.
    paths=[root/f'invocations_per_function_md.anon.d{d:02d}.csv' for d in range(1,15)]
    missing=[str(p) for p in paths if not p.exists()]
    if missing: raise FileNotFoundError('All 14 invocation days required; missing: '+', '.join(missing))
    return paths

def read_daily(root, max_functions, min_invocations, min_active_ratio):
    paths=build_index(root)
    series={}
    triggers={}
    for day,path in enumerate(paths):
        log.info('Reading day %d/14: %s',day+1,path.name)
        for chunk in pd.read_csv(path,chunksize=2000):
            required=['HashFunction','Trigger']+[str(i) for i in range(1,1441)]
            absent=[c for c in required if c not in chunk.columns]
            if absent: raise ValueError(f'{path}: missing columns {absent[:10]}')
            # Avoid duplicate HashFunction rows within a day silently overwriting data.
            for _,row in chunk.iterrows():
                fid=str(row['HashFunction'])
                if fid=='nan': continue
                trigger=str(row['Trigger']).lower()
                # Only HTTP-triggered functions. Trigger is metadata, not inferred from counts.
                if 'http' not in trigger: continue
                if fid not in series:
                    series[fid]=np.zeros(14*1440,dtype=np.float32)
                    triggers[fid]=set()
                triggers[fid].add(trigger)
                vals=pd.to_numeric(row[[str(i) for i in range(1,1441)]],errors='coerce').fillna(0).to_numpy(dtype=np.float32)
                if np.any(vals<0): raise ValueError(f'Negative invocation counts for {fid}, day {day+1}')
                series[fid][day*1440:(day+1)*1440]+=vals
    eligible=[]
    for fid,ts in series.items():
        if float(ts.sum()) >= min_invocations and float(np.count_nonzero(ts))/len(ts)>=min_active_ratio:
            eligible.append(fid)
    eligible.sort()
    log.info('HTTP functions=%d; eligible=%d; selected=%d',len(series),len(eligible),min(max_functions,len(eligible)) if max_functions else len(eligible))
    if max_functions: eligible=eligible[:max_functions]
    return {fid:series[fid] for fid in eligible}

def feature_windows(fid,ts,extractor,window_size,stride):
    records=[]
    # Entirely within one day; avoids windows crossing temporal split boundaries.
    for day in range(14):
        offset=day*1440
        for start in range(0,1440-window_size+1,stride):
            absolute=offset+start
            f=extractor.extract_features(ts[absolute:absolute+window_size])
            f.update(function_hash=fid,window_idx=day*((1440-window_size)//stride+1)+start//stride,
                     window_start_minute=absolute,window_end_minute=absolute+window_size,
                     day=day,hour_of_day=start//60)
            records.append(f)
    return pd.DataFrame.from_records(records)

def main():
    a=args_parser()
    if a.smoke_test:
        a.max_functions=30
        a.checkpoint_dir=str(Path(a.checkpoint_dir).parent/'phase1_smoke_checkpoints')
        a.output_dir=str(Path(a.output_dir)/'phase1_smoke')
    if not 1<=a.window_size<=1440 or a.stride<1: raise ValueError('Invalid window/stride')
    ck=Path(a.checkpoint_dir); ck.mkdir(parents=True,exist_ok=True)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    cfg={'window_size':a.window_size,'stride':a.stride,'min_invocations':a.min_invocations,
         'min_active_ratio':a.min_active_ratio,'max_functions':a.max_functions,
         'http_only':True,'days':14,'within_day_windows':True}
    manifest=ck/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old!=cfg: raise RuntimeError('Checkpoint settings changed. Choose a fresh --checkpoint-dir or restore original settings.')
        if not a.resume: raise RuntimeError('Checkpoint directory already exists. Use --resume or a fresh --checkpoint-dir.')
    else: manifest.write_text(json.dumps(cfg,indent=2))
    data=read_daily(Path(a.data_dir),a.max_functions,a.min_invocations,a.min_active_ratio)
    if not data: raise RuntimeError('No eligible HTTP functions. For smoke test consider --min-active-ratio 0.0')
    extractor=TimeSeriesFeatureExtractor()
    parts=[]
    for idx,(fid,ts) in enumerate(tqdm(data.items(),desc='Extracting functions')):
        path=ck/f'function_{idx:06d}.pkl'
        if a.resume and path.exists():
            df=pd.read_pickle(path)
            if not (df['function_hash']==fid).all(): raise RuntimeError(f'Checkpoint identity mismatch: {path}')
        else:
            df=feature_windows(fid,ts,extractor,a.window_size,a.stride)
            temp=path.with_suffix('.tmp')
            df.to_pickle(temp)
            os.replace(temp,path)
        parts.append(path)
    log.info('Finished feature extraction; %d function checkpoints',len(parts))
    if a.features_only: return
    # Label in batches to avoid holding all raw feature frames and labeled copies simultaneously.
    labeler=WindowedWeakSupervisionLabeler()
    labeled_parts=[]
    for path in tqdm(parts,desc='Labeling functions'):
        dest=path.with_suffix('.labeled.pkl')
        if a.resume and dest.exists():
            labeled=pd.read_pickle(dest)
        else:
            labeled=labeler.label_windows(pd.read_pickle(path))
            tmp=dest.with_suffix('.tmp'); labeled.to_pickle(tmp); os.replace(tmp,dest)
        labeled_parts.append(labeled)
    full=pd.concat(labeled_parts,ignore_index=True)
    del labeled_parts
    log.info('Rows=%d, label distribution=%s',len(full),full['label'].value_counts(dropna=False).to_dict())
    # zero-based day: 0-8 train, 9-10 validation, 11-13 test
    train=full[full.day.between(0,8)].copy()
    val=full[full.day.between(9,10)].copy()
    test=full[full.day.between(11,13)].copy()
    if min(map(len,(train,val,test)))==0: raise RuntimeError(f'Empty chronological split: {[len(train),len(val),len(test)]}')
    log.info('Chronological split: train=%d val=%d test=%d',len(train),len(val),len(test))
    for name,df in [('train',train),('val',val),('test',test)]:
        log.info('%s labels: %s',name,df.label.value_counts(dropna=False).to_dict())
        if df.label.isna().any(): raise RuntimeError(f'{name} has unlabeled rows; inspect labeling rules')
    labels=set(train.label)
    if len(labels)<2: raise RuntimeError('Fewer than 2 training classes; expand sample or audit labeling')
    for name,df in [('val',val),('test',test)]:
        unseen=set(df.label)-labels
        if unseen: raise RuntimeError(f'{name} contains unseen labels: {unseen}')
    # Ensure all features are numeric and finite before fitting.
    cols=[c for c in train.columns if c not in META+['label','confidence']]
    for name,df in [('train',train),('val',val),('test',test)]:
        bad=[c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        if bad: raise RuntimeError(f'{name} nonnumeric features: {bad}')
        df[cols]=df[cols].replace([np.inf,-np.inf],np.nan).fillna(0)
    trainer=WorkloadClassifierTrainer(output_dir=str(out))
    Xtr,ytr,Xv,yv,Xte,yte=trainer.prepare_data(train,val,test)
    trainer.train_lightgbm(Xtr,ytr,Xv,yv,use_gpu=False)
    result=trainer.evaluate_model('lightgbm',Xte,yte)
    log.info('Accuracy: %.5f; avg confidence: %.5f',result['classification_report']['accuracy'],result['avg_confidence'])
    trainer.plot_confusion_matrix(result,'lightgbm',save_path=str(out/'confusion_matrix_lightgbm.png'))
    trainer.plot_feature_importance('lightgbm',top_n=15,save_path=str(out/'feature_importance_lightgbm.png'))
    trainer.save_models()
    # Phase 2 expects this name; preserve the original trained-model filename too.
    shutil.copy2(out/'lightgbm_model.pkl', out/'lightgbm_classifier.pkl')
    (out/'phase1_metrics.json').write_text(json.dumps({'report':result['classification_report'],
        'avg_confidence':float(result['avg_confidence']),
        'confusion_matrix':result['confusion_matrix'].tolist(),
        'labels':list(result['labels']),
        'splits':{'train':len(train),'validation':len(val),'test':len(test)},'settings':cfg},indent=2,default=str))
    log.info('Phase 1 complete. Saved artifacts to %s',out)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    main()
