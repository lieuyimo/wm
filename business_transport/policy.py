"""Fair bounded candidate rollouts for five business scheduling baselines."""
from __future__ import annotations

import time
from dataclasses import asdict
from experiments.service_value.scoring import workload_score
from business_transport.dispatch import BusinessCPSAT
from business_transport.dynamics import CapacityDynamics,WorkflowDynamics
from business_transport.workflow_scoring import workflow_components,score_components
from business_transport.workflow_transition import check_business_transition
from lab_transport.decision_value import batch_candidates,candidates_for,nearest_batch
from lab_transport.paper_reuse.contracts import check_transition


class BusinessPolicy:
    def __init__(self,method,geometry,rules,observer,model=None,*,seed=0,horizon=5,
                 particles=3,budget=4,solver_seconds=.5,workflow_prediction=False,scoring='legacy',
                 capture_traces=False):
        if method not in ('nearest','nearest_rollout','cpsat_rule','cpsat_pets','cpsat_structured'):
            raise ValueError('unknown business scheduling baseline')
        self.method,self.geometry,self.rules,self.observer=method,geometry,rules,observer
        self.seed,self.horizon,self.particles,self.budget=seed,horizon,particles,budget
        if scoring not in ('legacy','workflow'):raise ValueError('unknown business score')
        self.workflow_prediction,self.scoring=workflow_prediction,scoring
        self.capture_traces=capture_traces
        adapter=WorkflowDynamics if workflow_prediction else CapacityDynamics
        self.model=adapter(model,rules) if model is not None else None
        self.solver=(BusinessCPSAT(geometry,rules,candidate_limit=budget,time_limit_seconds=solver_seconds,seed=seed)
                     if method.startswith('cpsat_') else None)
        self.log=[];self.fallbacks=0;self.rollout_steps=0
        self.prediction_calls=0;self.prediction_wall_seconds=0.0

    def candidates(self,observation,candidates):
        if self.method=='nearest':return (nearest_batch(observation,candidates,self.geometry),)
        if self.method=='nearest_rollout':
            # Existing nearest-ranked compatible matchings; empty wait action
            # is not in the CP-SAT action space, so it is excluded here too.
            plans,_=batch_candidates(observation,candidates,self.geometry,self.budget+1)
            plans=tuple(x for x in plans if x)
            return plans or ((),)
        return self.solver.proposals(observation,candidates)

    def evaluate(self,observation,plans):
        records=[]
        for plan in plans:
            outcomes=[];components=[];particles=[]
            for particle in range(self.particles):
                particle_seed=self.seed+observation.tick*1009+particle
                self.model.reset_seed(particle_seed)
                state=observation;trace=[];captured=[]
                for step in range(self.horizon):
                    actions=plan if step==0 else nearest_batch(state,candidates_for(state),self.geometry)
                    started=time.perf_counter()
                    try:
                        nxt=self.model.predict_next(state,actions)
                    finally:
                        self.prediction_calls+=1;self.prediction_wall_seconds+=time.perf_counter()-started
                    if self.workflow_prediction:check_business_transition(state,nxt,self.geometry,self.rules,actions)
                    else:check_transition(state,nxt,self.geometry)
                    if self.capture_traces:
                        prior_ids={s.id for s in state.samples}
                        captured.append({'tick':nxt.tick,'actions':[asdict(a) for a in actions],
                            'robots':[asdict(r) for r in nxt.robots],
                            'samples':[asdict(s) for s in nxt.samples],
                            'stations':[asdict(s) for s in nxt.stations],
                            'successor_releases':[s.id for s in nxt.samples if s.id not in prior_ids]})
                    self.rollout_steps+=1;trace.append(nxt);state=nxt
                if self.capture_traces:particles.append({'particle':particle,'seed':particle_seed,'steps':captured})
                if self.scoring=='workflow':
                    c=workflow_components(observation,trace,self.geometry,self.rules)
                    outcomes.append(score_components(c));components.append(c)
                else:
                    outcomes.append(workload_score(observation,trace,self.geometry))
            score=tuple(sum(s[i] for s in outcomes)/len(outcomes) for i in range(3))
            record={'assignments':[{'robot_id':a.robot_id,'sample_id':a.sample_id} for a in plan],
                    'score':score}
            if self.scoring=='workflow':record['components']=components
            if self.capture_traces:record['prediction_particles']=particles
            records.append(record)
        return records

    def __call__(self,observation,candidates):
        started=time.perf_counter()
        obs=self.observer.observe(observation)
        plans=self.candidates(obs,candidates)
        records=[];reason=None;detail=None
        before=getattr(getattr(self.model,'distribution',None),'forward_calls',0)
        service_before=getattr(getattr(self.model,'model',None),'service_calls',0)
        try:
            if self.method=='nearest':choice=0
            else:
                records=self.evaluate(obs,plans)
                choice=min(range(len(plans)),key=lambda i:(records[i]['score'],i))
        except (ValueError,RuntimeError,TypeError,KeyError,StopIteration,OverflowError) as exc:
            self.fallbacks+=1;reason=type(exc).__name__;detail=str(exc)[:300];choice=0
        after=getattr(getattr(self.model,'distribution',None),'forward_calls',0)
        entry={'tick':obs.tick,'candidate_count':len(candidates),'plans':records,
            'selected':[{'robot_id':a.robot_id,'sample_id':a.sample_id} for a in plans[choice]],
            'fallback':reason,'model_forward_calls':after-before,
            'structured_service_calls':getattr(getattr(self.model,'model',None),'service_calls',0)-service_before,
            'wall_seconds':time.perf_counter()-started}
        if self.capture_traces:
            entry.update(legal_candidates=[asdict(a) for a in candidates],
                         candidate_batches=[[asdict(a) for a in plan] for plan in plans],selected_index=choice,
                         fallback_detail=detail)
            if self.solver and self.solver.log:
                entry.update(solver_statuses=self.solver.log[-1].get('statuses',[]),
                             solver_fallback_reason=self.solver.log[-1].get('fallback'),
                             solver_truncated=self.solver.log[-1].get('partial_truncation',False))
        self.log.append(entry)
        return plans[choice]


class WorkflowPolicy(BusinessPolicy):
    """Business-facing entry point with full known-workflow rollout enabled.

    BusinessPolicy's original defaults remain available for archived baselines.
    Supply the same rule/PETS/structured movement adapter used by execution.
    """
    def __init__(self,method,geometry,rules,observer,model=None,**kwargs):
        if 'workflow_prediction' in kwargs or 'scoring' in kwargs:
            raise ValueError('WorkflowPolicy always uses workflow prediction and score')
        super().__init__(method,geometry,rules,observer,model,
                         workflow_prediction=True,scoring='workflow',**kwargs)
