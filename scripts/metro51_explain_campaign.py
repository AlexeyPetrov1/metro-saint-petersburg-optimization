"""Explain committed journals independently of the simulation process."""

import argparse
import json
from pathlib import Path
from time import monotonic,sleep

from metro51_explanations import explain_decision,load_env,validate_explanation
from metro51_run import atomic_json,journal


def process_packets(root,*,provider=None,timeout_s=15,cache=None):
    root=Path(root);done,errors=0,0
    for path in sorted(root.glob('*/explanations/facts_*.json')):
        out=path.with_name(path.name.replace('facts_','explanation_'))
        packet=json.loads(path.read_text(encoding='utf-8'))
        if out.exists():
            saved=json.loads(out.read_text(encoding='utf-8'))
            if saved.get('metadata',{}).get('error_code')!='disabled':
                validate_explanation(packet,saved)
                continue
        answer=explain_decision(packet,provider=provider,timeout_s=timeout_s,cache=cache)
        validate_explanation(packet,answer)
        atomic_json(out,answer)
        meta=answer['metadata'];errors+=meta['source']!='yandexgpt';done+=1
        journal(root/'logs/explanations.jsonl',dict(arm=path.parent.parent.name,decision_id=packet['decision_id'],
            source=meta['source'],error_code=meta['error_code'],seconds=meta['elapsed_seconds'],
            usage=meta.get('usage'),request_count=meta.get('request_count')))
    return dict(processed=done,fallbacks=errors)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--timeout',type=float,default=15)
    parser.add_argument('--max-seconds',type=float,default=10800)
    args=parser.parse_args();load_env();started=monotonic();cache={}
    total=0
    while True:
        result=process_packets(args.root,timeout_s=args.timeout,cache=cache);total+=result['processed']
        print(json.dumps(dict(stage='explained',**result,total=total),ensure_ascii=False),flush=True)
        complete=all((args.root/arm/'result.json').exists() for arm in ('reference','old','ensemble_old','balanced'))
        if not args.watch or complete or monotonic()-started>=args.max_seconds:
            break
        sleep(5)
    atomic_json(args.root/'logs/explanations.status.json',dict(total=total,all_simulations_finished=complete,
        elapsed_seconds=monotonic()-started,ended=True))


if __name__=='__main__':
    main()
