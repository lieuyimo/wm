"""Verify the fixed system demo using its exported data, not the webpage labels."""
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_run(root):
    data=read(root/'run.json')
    assert read(root/'summary.json')==data['business']['summary']
    assert read(root/'decisions.json')['decisions']==data['decisions']
    assert [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]==data['events']
    assert (root/'report.html').is_file()
    summary=data['business']['summary']
    assert not summary['resource']['violations']
    assert not summary['discrete_safety_violations']
    assert not summary['duplicate_assignments']
    assert not summary['lifecycle_violations']
    assert all(summary['motion_safety'][key]==0 for key in ('vertex_collisions','edge_swaps','invalid_edges'))
    assert 'service_remaining' not in json.dumps(data['history'])
    for decision in data['decisions']:
        assert len(decision['legal_candidates'])==decision['candidate_count']
        for plan in decision['plans']:
            assert 'service_remaining' not in json.dumps(plan.get('prediction_particles',[]))
    return data


def main(root):
    root=Path(root)
    comparison=read(root/'comparison'/'comparison.json')
    assert comparison['reference_policy']=='nearest'
    models={'nearest':'none','workflow_rule':'rule','workflow_pets':'pets','workflow_structured':'structured'}
    for policy,model in models.items():
        data=verify_run(root/'comparison'/policy)
        assert data['business']['model_name']==model
        assert comparison['methods'][policy]==data['business']['summary']
        assert data['business']['summary']['jobs_completed']==data['business']['summary']['jobs']==6
    assert comparison['methods']['nearest']['prediction_calls']==0
    assert comparison['methods']['workflow_pets']['pets_forward_calls']>0
    assert comparison['methods']['workflow_structured']['structured_service_calls']>0
    for horizon in ('1','3','5'):
        stats=comparison['independent_factual_prediction']['horizons'][horizon]['dynamic']
        assert len({stats[name]['windows'] for name in ('rule','pets','structured')})==1
    clinic=verify_run(root/'clinic')
    summary=clinic['business']['summary']
    assert (summary['jobs_completed'],summary['jobs'],clinic['summary']['ticks'])==(5,5,115)
    assert (summary['lacam3_triggers'],summary['lacam3_successes'],summary['lacam3_cache_steps'])==(2,2,11)
    three=verify_run(root/'three-stage')
    assert (three['business']['summary']['jobs_completed'],three['summary']['completed_count'])==(1,3)
    assert [e['sample_id'] for e in three['events'] if e['type']=='stage_ready']==['J1::stage2','J1::stage3']
    outage=verify_run(root/'outage')
    assert outage['business']['summary']['resource']['capacity_wait_ticks']>0
    assert outage['business']['summary']['station_unavailable_wait_ticks']>0
    assert any(e['type']=='service_paused_observed' for e in outage['events'])
    print('Verified exported data: four strategies, common factual windows, three stages, outage and clinic recovery')


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('usage: python -m experiments.system_demo.verify OUTPUT_DIRECTORY')
    main(sys.argv[1])
