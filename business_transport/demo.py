"""Run and inspect the existing multi-robot workflow system on CPU.

Predictions are captured from the actual policy rollout; no factual future is
fed back to the scheduler. All supplied workloads are simulation assumptions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(f'缺少依赖 {exc.name}；请安装 requirements-paper-reuse.lock') from exc

from business_transport.lacam3_integration import HybridBusinessSimulator,RecoveryMovementModel,RecoveryObserver
from business_transport.pibt_integration import MotionObserver,PIBTMovementModel
from business_transport.pibt_motion import PIBTMotion
from business_transport.policy import BusinessPolicy,WorkflowPolicy
from business_transport.visualization import render_business_html,render_comparison_html
from experiments.business_constraints.benchmark import make_case,read_config
try:
    from experiments.business_constraints.run import CHECKPOINT,make_model,summarize_result
    from experiments.service_value.counterfactual import CHECKPOINT_DIR,load_pr9_model
    from experiments.workflow_prediction.evaluate import evaluate_state_windows,rich_rows
except ModuleNotFoundError as exc:
    raise SystemExit(f'缺少依赖 {exc.name}；请安装 requirements-paper-reuse.lock') from exc
from experiments.pibt_motion.benchmark import safety
from experiments.pibt_motion.hospital_reproduction import load_bases,pilot_case
from experiments.workflow_prediction.config_audit import CONFIG as WORKFLOW_CONFIG
from lab_transport.dynamics_model import Geometry
from lab_transport.paper_reuse.ortools_dispatch import cp_model

POLICIES={'nearest':None,'workflow_rule':'rule','workflow_pets':'pets','workflow_structured':'structured'}
PETS_SHA='3748a5b96af507c1708355b6b33c25d1ecb56813a57b8469fc9144650bf35e38'
PIBT_SHA='faab5b916649549f1cd563df8dbf6e4f6382f631'
LACAM_SHA='1c3ecf8ade40298f16047544634e9c0c1cf3b93c'


def workflow_settings():
    return json.loads(WORKFLOW_CONFIG.read_text(encoding='utf-8'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _version(name):
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:return None


def _repo_revision():
    try:
        return subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):return None


def _repo_dirty():
    try:
        return bool(subprocess.check_output(['git','status','--porcelain'],stderr=subprocess.DEVNULL,text=True).strip())
    except (OSError,subprocess.CalledProcessError):return None


def empty_directory(path):
    path=Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'输出目录非空，拒绝覆盖：{path}')
    path.mkdir(parents=True,exist_ok=True)
    return path


def require_native(motion,pibt,lacam):
    if motion in ('pibt','hybrid') and (not pibt or not Path(pibt).is_file()):
        raise FileNotFoundError('PIBT 原生库不存在；使用 --pibt-library 或 WM_PIBT_LIBRARY 指定已构建文件')
    if motion=='hybrid' and (not lacam or not Path(lacam).is_file() or not os.access(lacam,os.X_OK)):
        raise FileNotFoundError('LaCAM3 可执行文件不存在或不可执行；使用 --lacam-worker 或 WM_LACAM3_WORKER')


def build_case(scenario,seed,regime,source_dir):
    if scenario in ('empty8','random32'):
        configured=read_config()
        return (*make_case(configured,scenario,seed,regime),{'kind':'pinned_map_and_pr15_workload',
                'map_sha256':configured['layouts'][scenario]['sha256']})
    if scenario=='clinic':
        if not source_dir:
            raise ValueError('clinic 需要 --source-dir 或 WM_HOSPITAL_SOURCES，包含四份已校验的公开源文件')
        bases,hashes=load_bases(source_dir);business,config=pilot_case('clinic',bases['clinic'])
        return business,config,{'kind':'pr14_archived_clinic','source_sha256':hashes,
                                'note':'复现原活锁请使用 nearest、hybrid、seed=61000'}
    if scenario=='three_stage':
        # Reuse the exact deterministic PR15 test fixture; no new task generator.
        from experiments.workflow_prediction.tests.test_workflow import case
        business=case(3)
        return business,{'source':'experiments.workflow_prediction.tests.test_workflow.case(3)'},\
               {'kind':'pr15_deterministic_three_stage_fixture'}
    raise ValueError('unknown scenario')


def model_for(name,geometry,rules,motion,pibt,lacam,planner_seed):
    if name is None:return None,None
    if cp_model is None:raise RuntimeError('OR-Tools CP-SAT 不可用；请安装 requirements-paper-reuse.lock')
    network=info=None
    if name=='structured':
        checkpoint=CHECKPOINT_DIR/'sequence.pt'
        if not checkpoint.is_file():raise FileNotFoundError(f'结构化模型检查点不存在：{checkpoint}')
        network,info=load_pr9_model()
        info['checkpoint']=str(checkpoint)
    elif name=='pets':
        if not CHECKPOINT.is_file():raise FileNotFoundError(f'PETS 检查点不存在：{CHECKPOINT}')
        if _sha(CHECKPOINT)!=PETS_SHA:raise ValueError('PETS 检查点哈希不符，拒绝替换模型')
        info={'checkpoint':str(CHECKPOINT),'checkpoint_sha256':_sha(CHECKPOINT)}
    base=make_model(name,geometry,network,info['threshold'] if name=='structured' else None,planner_seed)
    if name=='rule':info={'kind':'ProgressAwareRule','checkpoint':None}
    if motion=='hybrid':model=RecoveryMovementModel(base,geometry,rules,pibt,lacam,planner_seed=planner_seed)
    elif motion=='pibt':model=PIBTMovementModel(base,geometry,PIBTMotion(geometry,pibt,planner_seed=planner_seed))
    else:model=base
    return model,info


def preflight_models(names,geometry,rules,motion,pibt,lacam,planner_seed):
    """Fail before creating results if a required solver/model cannot load."""
    if motion in ('pibt','hybrid'):
        PIBTMotion(geometry,pibt,planner_seed=planner_seed)
    for name in names:
        if name is not None:model_for(name,geometry,rules,motion,pibt,lacam,planner_seed)


class ObservedSimulator(HybridBusinessSimulator):
    """Record calls and executed steps around the existing motion backend."""
    def __init__(self,*args,**kwargs):
        self.executed_motion=[]
        super().__init__(*args,**kwargs)

    def _move(self,tick,serviced):
        before={rid:r.position for rid,r in self.robots.items()}
        calls=self.pibt.solver_calls if self.pibt else 0
        paused={rid for rid,intervals in self.motion_pauses.items() if any(a<=tick<b for a,b in intervals)}
        super()._move(tick,serviced)
        self.executed_motion.append({'tick':tick,'backend':('hybrid' if self.hybrid_enabled else self.motion_backend),
            'pibt_calls':(self.pibt.solver_calls-calls) if self.pibt else 0,
            'locked':sorted(serviced|paused),'before':before,
            'after':{rid:r.position for rid,r in self.robots.items()}})


def _history_frame(observation,previous,stationary):
    robots=[]
    for robot in observation.robots:
        last=previous.get(robot.id)
        stationary[robot.id]=(stationary.get(robot.id,0)+1 if last==robot.position else 0)
        previous[robot.id]=robot.position
        value=asdict(robot);value['stationary_steps']=stationary[robot.id];robots.append(value)
    return {'tick':observation.tick,'robots':robots,'samples':[asdict(s) for s in observation.samples],
            'stations':[asdict(s) for s in observation.stations]}


def _observed_events(result,sim,frames,decisions,model_name):
    events=list(result['events'])
    for decision in decisions:
        events.append({'tick':decision['tick'],'type':'dispatch_decision','model':model_name,
                       'selected':decision['selected'],'fallback':decision['fallback'],
                       'fallback_detail':decision.get('fallback_detail'),
                       'solver_fallback_reason':decision.get('solver_fallback_reason')})
    for row in sim.executed_motion:
        if row['pibt_calls']:
            events.append({'tick':row['tick'],'type':'pibt_step','calls':row['pibt_calls'],
                           'before':row['before'],'after':row['after'],'locked':row['locked']})
    for row in sim.recovery_log:
        if row['triggered'] or row['cache_used'] or row['cache_invalidated']:
            events.append({'type':'lacam3_recovery',**row})
    old_devices={};old_paused=set()
    for frame in frames:
        devices={s['id']:s.get('device_available',s['available']) for s in frame['stations']}
        for sid,value in devices.items():
            if sid in old_devices and old_devices[sid]!=value:
                events.append({'tick':frame['tick'],'type':'device_status_observed','station_id':sid,'available':value})
        old_devices=devices
        stations={tuple(s['position']):s for s in frame['stations']}
        paused={(r['id'],stations[tuple(r['position'])]['id']) for r in frame['robots']
                if tuple(r['position']) in stations and r['task_id'] is not None and r['service_elapsed'] is not None
                and not stations[tuple(r['position'])].get('device_available',True)}
        for rid,sid in sorted(paused-old_paused):
            events.append({'tick':frame['tick'],'type':'service_paused_observed','robot_id':rid,'station_id':sid,
                           'source':'current observable service and device status'})
        for rid,sid in sorted(old_paused-paused):
            events.append({'tick':frame['tick'],'type':'service_pause_ended_observed','robot_id':rid,'station_id':sid,
                           'source':'current observable service and device status'})
        old_paused=paused
    return sorted(events,key=lambda e:e['tick'])


def run_case(business,config,source,*,scenario,regime,policy_name,model_name,motion,seed,pibt,lacam,
             horizon,particles,budget,planner_seed):
    torch.set_num_threads(1);geometry=Geometry(business.scenario.grid)
    observer=RecoveryObserver() if motion=='hybrid' else MotionObserver()
    model,model_info=model_for(model_name,geometry,business.rules,motion,pibt,lacam,planner_seed)
    if policy_name=='nearest':
        policy=BusinessPolicy('nearest',geometry,business.rules,observer,seed=planner_seed,capture_traces=True)
    else:
        policy=WorkflowPolicy('cpsat_'+model_name,geometry,business.rules,observer,model,seed=planner_seed,
                              horizon=horizon,particles=particles,budget=budget,
                              solver_seconds=workflow_settings()['cp_sat_time_limit_seconds'],capture_traces=True)
    sim=ObservedSimulator(business,'world',seed,motion_backend=motion,pibt_library=pibt,
                          lacam3_worker=lacam,planner_seed=planner_seed,collect_metrics=True)
    rows=[];frames=[];previous={};stationary={}
    def observed(raw):
        frames.append(_history_frame(observer.observe(raw),previous,stationary))
    result=sim.run(dispatch_policy=policy,on_observation=observed,on_transition=rows.append)
    if rows and (not frames or frames[-1]['tick']!=rows[-1].next_observation.tick):
        frames.append(_history_frame(observer.observe(rows[-1].next_observation),previous,stationary))
    metrics=summarize_result(result,business,policy)
    metrics.pop('decisions',None);metrics.pop('solver_log',None)
    metrics['prediction_calls']=policy.prediction_calls
    metrics['prediction_wall_ms']=policy.prediction_wall_seconds*1000
    metrics['motion_safety']=safety(result['history'],geometry)
    metrics['pibt_calls']=sim.pibt.solver_calls if sim.pibt else 0
    metrics['lacam3_triggers']=sum(e['triggered'] for e in sim.recovery_log)
    metrics['lacam3_solver_calls']=sum(bool(e.get('worker_invoked')) for e in sim.recovery_log)
    metrics['lacam3_successes']=sum(e.get('status')=='solved' for e in sim.recovery_log)
    metrics['lacam3_cache_steps']=sum(e['cache_used'] for e in sim.recovery_log)
    metrics['lacam3_fallback_statuses']=[e['status'] for e in sim.recovery_log
                                         if e['triggered'] and e['status']!='solved']
    events=_observed_events(result,sim,frames,policy.log,model_name)
    static_scenario={'stations':{sid:asdict(station) for sid,station in business.scenario.stations.items()},
        'initial_robots':[asdict(r) for r in business.scenario.robots],
        'sample_specs':[asdict(s) for s in business.scenario.samples],
        'duration_jitter':business.scenario.duration_jitter,'max_ticks':business.scenario.max_ticks,
        'note':'complete offline scenario configuration; the online model receives only current observations and public workflows'}
    provenance={'repository_commit':_repo_revision(),'repository_dirty':_repo_dirty(),'python':platform.python_version(),
        'platform':platform.platform(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,
        'ortools':_version('ortools'),'mbrl':_version('mbrl'),
        'mbrl_source_commit':'3f93cccfc8d635f74e335a2f07aab6e9a48fc021',
        'pibt_expected_source_commit':PIBT_SHA if motion in ('pibt','hybrid') else None,
        'pibt_binary_sha256':_sha(pibt) if motion in ('pibt','hybrid') else None,
        'lacam3_expected_source_commit':LACAM_SHA if motion=='hybrid' else None,
        'lacam3_binary_sha256':_sha(lacam) if motion=='hybrid' else None,'model':model_info}
    output={'schema':'wm-integrated-demo-v1','scenario':result['scenario'],'policy':policy_name,'seed':seed,
        'summary':result['summary'],'samples':result['samples'],'history':frames,'events':events,
        'decisions':policy.log,'business':{'jobs':list(business.jobs),'workflows':business.rules.workflows,
            'resources':[asdict(r) for r in business.rules.resources],'summary':metrics,
            'model_name':model_name or 'none','motion_backend':motion,
            'assumptions':'公开地图上的合成任务、站点和服务时间；不代表真实实验室或物理安全认证。'},
        'configuration':{'scenario':scenario,'regime':regime,'scenario_config':config,'static_scenario':static_scenario,'source':source,
            'policy':policy_name,'model':model_name or 'none','motion_backend':motion,'environment_seed':seed,
            'planner_seed':planner_seed,'horizon':horizon,'particles':particles,'candidate_budget':budget,
            'workflow_protocol_path':str(WORKFLOW_CONFIG),'workflow_protocol_sha256':_sha(WORKFLOW_CONFIG),
            'cp_sat_time_limit_seconds':workflow_settings()['cp_sat_time_limit_seconds'],
            'prediction_capture':'actual online candidate rollouts; no factual future input',
            'motion_forecast_note':('legacy backend may use recovery not modeled by its rule predictor' if motion=='legacy'
                                    else 'execution and prediction use the same selected motion adapter')},
        'provenance':provenance}
    return output,rows


def write_run(output,root):
    root=Path(root)
    (root/'run.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (root/'summary.json').write_text(json.dumps(output['business']['summary'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (root/'events.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in output['events']),encoding='utf-8')
    (root/'decisions.json').write_text(json.dumps({'model':output['business']['model_name'],
            'decisions':output['decisions']},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (root/'report.html').write_text(render_business_html(output),encoding='utf-8')


def resolve_model(policy,requested):
    actual=POLICIES[policy]
    if requested!='auto' and requested!=(actual or 'none'):
        raise ValueError(f'策略 {policy} 要求模型 {actual or "none"}，不能静默替换为 {requested}')
    return actual


def comparison(args,root,business,config,source):
    if args.motion!='hybrid':raise ValueError('四组对照的共同事实轨迹评价目前仅支持 --motion hybrid')
    results={};versions={};run_configs={};reference_rows=None
    for policy_name in POLICIES:
        model_name=POLICIES[policy_name];outdir=empty_directory(root/policy_name)
        result,rows=run_case(business,config,source,scenario=args.scenario,regime=args.regime,
            policy_name=policy_name,model_name=model_name,motion=args.motion,seed=args.seed,
            pibt=args.pibt_library,lacam=args.lacam_worker,horizon=args.horizon,
            particles=args.particles,budget=args.budget,planner_seed=args.planner_seed)
        write_run(result,outdir);results[policy_name]=result['business']['summary']
        versions[policy_name]=result['provenance'];run_configs[policy_name]=result['configuration']
        if policy_name=='nearest':reference_rows=rows
        print(policy_name,results[policy_name]['jobs_completed'],'/',results[policy_name]['jobs'],flush=True)
    network,info=load_pr9_model()
    settings=workflow_settings()
    eval_config={'planner_seed':args.planner_seed,'prediction_horizons':settings['prediction_horizons'],
                 'prediction_window_stride':settings['prediction_window_stride']}
    accuracy=evaluate_state_windows(rich_rows(reference_rows),business,Geometry(business.scenario.grid),
        network,info['threshold'],args.pibt_library,args.lacam_worker,eval_config)
    result={'scope':'same scenario and keyed disturbances; forecast accuracy uses one common nearest factual trajectory',
        'reference_policy':'nearest','scenario':args.scenario,'regime':args.regime,'seed':args.seed,
        'motion_backend':args.motion,'horizon':args.horizon,'particles':args.particles,'candidate_budget':args.budget,
        'methods':results,'run_provenance':versions,'run_configuration':run_configs,
        'independent_factual_prediction':accuracy,
        'limitation':'the existing PR15 scenario is reused for system demonstration, not a new independent test'}
    (root/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (root/'comparison.html').write_text(render_comparison_html(result),encoding='utf-8')
    return result


def main(argv=None):
    settings=workflow_settings()
    default_case=settings['confirmation'][0]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','compare'))
    parser.add_argument('--scenario',choices=('empty8','random32','clinic','three_stage'),default=default_case['layout'])
    parser.add_argument('--regime',choices=tuple(read_config()['regimes']),default=default_case['regimes'][0])
    parser.add_argument('--seed',type=int,default=default_case['seed'])
    parser.add_argument('--policy',choices=tuple(POLICIES),default='workflow_rule')
    parser.add_argument('--model',choices=('auto','none','rule','pets','structured'),default='auto')
    parser.add_argument('--motion',choices=('legacy','pibt','hybrid'),default=settings['motion_backend'])
    parser.add_argument('--horizon',type=int,default=settings['horizon'])
    parser.add_argument('--particles',type=int,default=settings['particles'])
    parser.add_argument('--budget',type=int,default=settings['candidate_budget'])
    parser.add_argument('--planner-seed',type=int,default=settings['planner_seed'])
    parser.add_argument('--pibt-library',type=Path,default=os.environ.get('WM_PIBT_LIBRARY'))
    parser.add_argument('--lacam-worker',type=Path,default=os.environ.get('WM_LACAM3_WORKER'))
    parser.add_argument('--source-dir',type=Path,default=os.environ.get('WM_HOSPITAL_SOURCES'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.seed<0 or args.planner_seed<0 or min(args.horizon,args.particles,args.budget)<1:
            raise ValueError('种子不能为负，窗口、粒子数和候选预算必须为正')
        if args.command=='compare' and args.model!='auto':raise ValueError('compare 固定使用四种方案，请使用 --model auto')
        model=resolve_model(args.policy,args.model)
        require_native(args.motion,args.pibt_library,args.lacam_worker)
        if args.command=='compare' and cp_model is None:raise RuntimeError('OR-Tools CP-SAT 不可用')
        if args.command=='compare' and args.motion!='hybrid':raise ValueError('compare 的事实预测评价目前仅支持 hybrid')
        business,config,source=build_case(args.scenario,args.seed,args.regime,args.source_dir)
        preflight_models(POLICIES.values() if args.command=='compare' else (model,),
                         Geometry(business.scenario.grid),business.rules,args.motion,
                         args.pibt_library,args.lacam_worker,args.planner_seed)
        root=empty_directory(args.output)
        if args.command=='compare':comparison(args,root,business,config,source)
        else:
            result,_=run_case(business,config,source,scenario=args.scenario,regime=args.regime,
                policy_name=args.policy,model_name=model,motion=args.motion,seed=args.seed,
                pibt=args.pibt_library,lacam=args.lacam_worker,horizon=args.horizon,
                particles=args.particles,budget=args.budget,planner_seed=args.planner_seed)
            write_run(result,root)
            print(f"完成 {result['business']['summary']['jobs_completed']}/{result['business']['summary']['jobs']} 个业务任务；回放：{root/'report.html'}")
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError,ImportError,OSError) as exc:
        parser.exit(2,f'运行失败：{exc}\n')


if __name__=='__main__':main()
