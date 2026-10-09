import json

from metro51_explanations import build_fact_packet
from scripts.metro51_explain_campaign import process_packets


class Provider:
    model='test'
    def generate_structured(self,*,packet,timeout_s):
        return dict(run_id=packet['run_id'],decision_id=packet['decision_id'],context_hash=packet['context_hash'],
                    sections={k:['decision.action'] if k=='action' else []
                              for k in ('action','prediction','rationale','effects','risks','revisit')})


def test_worker_processes_committed_packets_once_without_changing_action(tmp_path):
    folder=tmp_path/'balanced'/'explanations';folder.mkdir(parents=True)
    packet=build_fact_packet(run_id='run',decision_id='1',action={'kind':'KEEP'},facts=[])
    path=folder/'facts_00001.json';path.write_text(json.dumps(packet),encoding='utf-8')
    before=path.read_bytes()
    assert process_packets(tmp_path,provider=Provider())['processed']==1
    answer=json.loads((folder/'explanation_00001.json').read_text(encoding='utf-8'))
    assert answer['action']==packet['action']
    assert answer['metadata']['source']=='yandexgpt'
    assert path.read_bytes()==before
    assert process_packets(tmp_path,provider=Provider())['processed']==0
