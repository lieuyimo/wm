"""Compare repeated fixed-seed results, excluding measurements and provenance.

Timing and native binary hashes vary with the runner; no behavior is omitted.
"""
import json
from pathlib import Path
import sys

IGNORE={'wall_seconds','decision_wall_ms','decision_wall_mean_ms','decision_wall_p95_ms',
        'prediction_wall_ms','planning_wall_ms','planning_cpu_ms','planning_peak_rss_kb','provenance',
        'run_provenance'}


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k not in IGNORE}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


def main(left,right):
    left,right=Path(left),Path(right)
    for name in ('comparison/nearest','comparison/workflow_rule','comparison/workflow_pets',
                 'comparison/workflow_structured','clinic','three-stage','outage'):
        a=clean(json.loads((left/name/'run.json').read_text()))
        b=clean(json.loads((right/name/'run.json').read_text()))
        if a!=b:raise AssertionError(f'fixed-seed behavior differs: {name}')
    a=clean(json.loads((left/'comparison/comparison.json').read_text()))
    b=clean(json.loads((right/'comparison/comparison.json').read_text()))
    if a!=b:raise AssertionError('fixed-seed factual evaluation differs')
    print('Repeated fixed-seed exported behavior and factual evaluation match; wall time and provenance excluded')


if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('usage: compare_runs.py LEFT_DIRECTORY RIGHT_DIRECTORY')
    main(*sys.argv[1:])
