#!/usr/bin/env python3
"""Train encoder ablations on one fixed split; synthetic results are labelled.

No policy is certified by this experiment. Latency covers warm single-prefix
model forwards, including all selected/full/complement paths, not tool execution.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from agent1.dataset import prepare_dataset, PrefixDataset, collate_rows
from agent1.environments import generate_demo
from agent1.registry import ToolRegistry
from agent1.engine import train, evaluate
from agent1.runtime import Ensemble
from agent1.io import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data',type=Path)
    source.add_argument('--synthetic',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--variants',nargs='+',default=['gru','loop1','loop2','loop4','no_time','temporal_only','graph_only'])
    args=parser.parse_args()
    definitions={'gru':{'encoder':'gru'},'loop1':{'refinement_steps':1},'loop2':{'refinement_steps':2},
                 'loop4':{'refinement_steps':4},'no_time':{'node_time_weighting':False},
                 'temporal_only':{'ablation':'temporal_only'},'graph_only':{'ablation':'graph_only'}}
    if set(args.variants)-set(definitions): parser.error('unknown variant')
    if args.output.exists() and any(args.output.iterdir()): parser.error('output must be empty')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    if args.synthetic:
        raw=args.output/'raw';data=args.output/'data'
        registry=ToolRegistry.from_json(ROOT/'configs/tool_registry.json')
        generate_demo(raw,registry,groups=16,formation=True)
        prepare_dataset(raw/'episodes.jsonl',raw/'registry.json',data)
    else:
        data=args.data
    dataset=PrefixDataset(data,'test'); registry=dataset.registry
    probe=collate_rows([dataset.rows[0]],registry)
    records=[]
    for name in args.variants:
        config={'hidden_dim':24,'edge_dim':8,'graph_layers':1,'encoder':'loop_transformer',**definitions[name]}
        started=time.perf_counter()
        report=train(data,args.output/name,seed=101,config={'epochs':args.epochs,'batch_size':8,'threads':1},model_config=config)
        training_seconds=time.perf_counter()-started
        ensemble=Ensemble([report['checkpoint']],registry)
        metrics=evaluate(ensemble,data,'test',args.output/name/'test.json')['metrics']
        model=ensemble.models[0]
        latencies=[]
        with torch.inference_mode():
            for _ in range(2): model(probe)
            for _ in range(12):
                now=time.perf_counter();model(probe);latencies.append((time.perf_counter()-now)*1000)
        record={'variant':name,'synthetic':dataset.manifest['synthetic'],'model_config':model.config.to_dict(),
                'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':training_seconds,
                'warm_forward_median_ms':statistics.median(latencies),'warm_forward_p95_ms':sorted(latencies)[-1],
                'risk_by_bin':metrics['risk_by_bin']['none'],'joint_readiness':metrics['formation_by_branch'].get('none'),
                'groups':len({r['tracking']['group_id'] for r in dataset.rows}),'seed':101}
        records.append(record);write_json(args.output/'comparison.json',{'scope':'engineering ablation; no production certification; p95 is nearest-rank over 12 warm samples','results':records})
    lines=['# Encoder comparison','',f'Synthetic: {dataset.manifest["synthetic"]}. One training seed; not evidence of real-world superiority.','',
           '| Variant | Parameters | Train seconds | Warm median ms | Warm p95 ms | Bin 1 Brier |',
           '|---|---:|---:|---:|---:|---:|']
    for r in records:
        brier=r['risk_by_bin']['1']['brier']
        lines.append(f'| {r["variant"]} | {r["parameters"]} | {r["training_seconds"]:.2f} | {r["warm_forward_median_ms"]:.2f} | {r["warm_forward_p95_ms"]:.2f} | {brier:.4f} |')
    (args.output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(args.output/'comparison.md')

if __name__=='__main__': main()
