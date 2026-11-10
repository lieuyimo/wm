"""Add business views to the existing self-contained timeline/map report."""
from __future__ import annotations

import html

from lab_transport.report import render_html

_CSS = r'''
.business-panel{margin-top:18px}.business-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
.business-panel h3{margin:18px 0 8px}.business-panel select{max-width:100%;padding:6px}
.business-panel pre{white-space:pre-wrap;word-break:break-word;background:#f3f7fb;padding:12px;border-radius:8px;font-size:12px;line-height:1.5}
.business-note{border-left:4px solid #1769aa;padding-left:10px;color:#405970}
.business-events{max-height:240px;overflow:auto;font-size:13px}.business-workflow{font-size:13px;line-height:1.6}
@media(max-width:760px){.business-grid{grid-template-columns:1fr}}
'''

_BODY = r'''
<section class="panel business-panel" id="business-panel">
 <h2>业务任务与资源状态</h2><p class="business-note" id="business-summary"></p>
 <div class="business-grid"><div><h3>机器人</h3><table><thead><tr><th>机器人</th><th>位置</th><th>当前阶段</th><th>连续未移动</th></tr></thead><tbody id="business-robots"></tbody></table>
 <h3>工作站与容量</h3><table><thead><tr><th>站点</th><th>设备</th><th>资源占用</th><th>服务准入</th></tr></thead><tbody id="business-stations"></tbody></table></div>
 <div><h3>已知样本的完整工作流</h3><div id="business-jobs" class="business-workflow"></div>
 <p class="foot">这里只显示当前已释放的业务任务；尚未释放的后续阶段来自该任务的公开流程模板。</p></div></div>
 <h2>当次调度与世界模型预测</h2><p id="business-decision" class="business-note"></p>
 <p>候选方案 <select id="forecast-candidate" aria-label="选择候选方案"></select>
 粒子 <select id="forecast-particle" aria-label="选择预测粒子"></select></p>
 <pre id="business-forecast"></pre>
 <p class="foot">预测是当时策略实际运行的模型输出。只有被执行的候选可以与事后实况并列；之后的调度动作与扰动可能不同，不能将差异视为单一候选的因果效果。</p>
 <h2>执行事件与恢复</h2><ul id="business-events" class="business-events"></ul>
 <p class="foot">“连续未移动”包括正常服务和排队，不直接判为异常；设备暂停由当前观测推断，恢复事件来自执行器日志。</p>
</section>
'''

_SCRIPT = r'''
const business=data.business;
const candidateSelect=document.getElementById('forecast-candidate'), particleSelect=document.getElementById('forecast-particle');
let currentBusinessDecision=null, previousBusinessTick=null;
const pos=p=>'('+p.join(', ')+')';
const taskNames={waiting:'待分配',assigned:'前往取样',in_transit:'运输中',completed:'已完成'};
function cellRow(body,values){const row=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=String(value);row.appendChild(td)}body.appendChild(row)}
function option(select,value,label){const item=document.createElement('option');item.value=String(value);item.textContent=label;select.appendChild(item)}
function summarizePrediction(state){
 const robots=state.robots.map(r=>r.id+'@'+pos(r.position)+' '+(r.task_id||'空闲')+'/'+r.phase).join('；');
 const tasks=state.samples.map(s=>s.id+':'+(taskNames[s.status]||s.status)).join('；')||'无';
 const stations=state.stations.map(s=>s.id+':' +(s.device_available?'正常':'不可用')+' '+s.resource_occupied+'/'+s.resource_capacity).join('；');
 return '机器人 '+robots+'\n任务 '+tasks+'\n站点 '+stations;
}
function renderForecast(){
 const output=document.getElementById('business-forecast');const d=currentBusinessDecision;
 if(!d){output.textContent='该时刻尚无调度决策。';return}
 if(business.model_name==='none'){output.textContent='最近优先不调用世界模型；无预测轨迹。';return}
 const index=Number(candidateSelect.value||0),record=d.plans[index];
 if(!record||!record.prediction_particles||!record.prediction_particles.length){output.textContent='本次预测未完成或已回退；没有可展示的模型轨迹。';return}
 const particle=record.prediction_particles[Math.min(Number(particleSelect.value||0),record.prediction_particles.length-1)];
 const chosen=index===d.selected_index;let text='候选 '+(index+1)+'；平均评分 '+JSON.stringify(record.score)+'；粒子 '+particle.particle+(chosen?'；已执行':'；未执行')+'\n';
 for(const step of particle.steps){
  text+='\n【预测 t='+step.tick+'】动作 '+JSON.stringify(step.actions)+'\n'+summarizePrediction(step)+'\n预测后续阶段释放 '+(step.successor_releases.join('、')||'无')+'\n';
  if(chosen){const actual=data.history.find(f=>f.tick===step.tick);
   text+=actual?'【事实 t='+step.tick+'，事后观测】\n'+summarizePrediction(actual)+'\n':'【事实】运行结束后没有对应观测。\n';
  }else text+='【事实】该候选未执行，没有其反事实观测。\n';
 }
 output.textContent=text;
}
candidateSelect.addEventListener('change',()=>{particleSelect.value='0';renderForecast()});
particleSelect.addEventListener('change',renderForecast);
function eventText(e){
 const labels={assigned:'任务分配',released:'阶段释放',stage_ready:'后继阶段就绪',picked_up:'已取样',delivered:'阶段送达',
  service_started:'服务开始',service_finished:'服务结束',resource_wait:'资源等待',pibt_step:'PIBT 协调移动',
  lacam3_recovery:'LaCAM3 恢复',dispatch_decision:'调度决策',device_status_observed:'设备状态变化',
  service_paused_observed:'观察到服务暂停',service_pause_ended_observed:'观察到服务暂停结束'};
 const bits=['t='+e.tick,labels[e.type]||e.type];
 for(const key of ['robot_id','sample_id','station_id','resource_id','status','fallback','invalidation_reason'])if(e[key]!==undefined&&e[key]!==null)bits.push(key+'='+e[key]);
 if(e.type==='lacam3_recovery'){if(e.triggered)bits.push('触发');if(e.cache_used)bits.push('执行缓存路径');if(e.plan_steps)bits.push('路径 '+e.plan_steps+' 步');if(e.detail)bits.push(e.detail)}
 return bits.join(' · ');
}
function renderBusiness(index){
 const frame=data.history[index],summary=business.summary;
 document.getElementById('business-summary').textContent='本次运行最终完成业务任务 '+summary.jobs_completed+'/'+summary.jobs+'；阶段 '+data.summary.completed_count+'/'+data.summary.sample_count+'；模型 '+business.model_name+'；运动 '+business.motion_backend+'。';
 const robots=document.getElementById('business-robots');robots.replaceChildren();
 for(const r of frame.robots)cellRow(robots,[r.id,pos(r.position),r.task_id||'空闲',r.stationary_steps+' 步']);
 const stations=document.getElementById('business-stations');stations.replaceChildren();
 for(const s of frame.stations)cellRow(stations,[s.id,s.device_available?'正常':'不可用',s.resource_id+' '+s.resource_occupied+'/'+s.resource_capacity,s.capacity_blocked?'容量等待':(s.available?'可服务':'暂停')]);
 const jobs=document.getElementById('business-jobs');jobs.replaceChildren();
 const known=[...new Set(frame.samples.map(s=>s.job_id).filter(Boolean))].sort();
 if(!known.length){const line=document.createElement('p');line.textContent='当前没有已释放的业务任务。';jobs.appendChild(line)}
 for(const id of known){const sample=frame.samples.find(s=>s.job_id===id),route=business.workflows[sample.workflow];
  const block=document.createElement('div'),title=document.createElement('strong');
  const final=frame.samples.find(s=>s.job_id===id&&s.stage_index===route.length-1&&s.status==='completed');
  const finalResult=final&&data.samples.find(s=>s.id===final.id);title.textContent=id+' · '+(final?'业务任务已完成，t='+finalResult.completed_tick:'业务任务进行中');block.appendChild(title);
  for(let i=0;i<route.length;i++){const s=frame.samples.find(x=>x.job_id===id&&x.stage_index===i);const line=document.createElement('div');
   let info='阶段 '+(i+1)+' '+route[i][0]+' → '+route[i][1]+'：';
   if(!s)info+='后续阶段，尚未释放';else{info+=taskNames[s.status]||s.status;
    if(s.status==='assigned'||s.status==='in_transit')info+=' · 机器人 '+s.robot_id;
    if(s.status==='completed'){const record=data.samples.find(x=>x.id===s.id);if(record&&record.completed_tick!==null&&record.completed_tick<=frame.tick)info+=' · 完成 t='+record.completed_tick}}
   line.textContent=info;block.appendChild(line)}jobs.appendChild(block)}
 const d=data.decisions.filter(x=>x.tick<=frame.tick).at(-1)||null;currentBusinessDecision=d;
 document.getElementById('business-decision').textContent=d?'t='+d.tick+' · 模型 '+business.model_name+' · 合法单项候选 '+d.candidate_count+'：'+JSON.stringify(d.legal_candidates)+' · 已选择 '+JSON.stringify(d.selected)+(d.solver_fallback_reason?' · 求解回退 '+d.solver_fallback_reason:'')+(d.fallback?' · 预测回退 '+d.fallback+' '+(d.fallback_detail||''):''):'该时刻尚无调度决策。';
 if(!d||previousBusinessTick!==d.tick){candidateSelect.replaceChildren();particleSelect.replaceChildren();
  if(d){for(let i=0;i<d.candidate_batches.length;i++){const r=d.plans[i];option(candidateSelect,i,'候选 '+(i+1)+' '+JSON.stringify(d.candidate_batches[i])+(r?' · 评分 '+JSON.stringify(r.score):''))}
   for(let i=0;i<(d.plans[0]?.prediction_particles?.length||0);i++)option(particleSelect,i,i);
   candidateSelect.value=String(d.selected_index);particleSelect.value='0'}previousBusinessTick=d?d.tick:null}
 renderForecast();
 const list=document.getElementById('business-events');list.replaceChildren();
 for(const event of data.events.filter(e=>e.tick<=frame.tick).slice(-18)){const item=document.createElement('li');item.textContent=eventText(event);list.appendChild(item)}
}
document.getElementById('subtitle').textContent=data.scenario.name+' · '+data.policy+' · 模型 '+business.model_name+' · 种子 '+data.seed;
document.getElementById('metrics').textContent='最终业务任务完成：'+business.summary.jobs_completed+'/'+business.summary.jobs+'；运输阶段：'+data.summary.completed_count+'/'+data.summary.sample_count+'；结束步数：'+data.summary.ticks;
'''


def render_business_html(result):
    page=render_html(result)
    page=page.replace('</style>',_CSS+'</style>',1)
    page=page.replace('<h2>任务状态</h2>','<h2>已释放的运输阶段</h2>',1)
    page=page.replace('</section></div>\n<script>','</section></div>'+_BODY+'\n<script>',1)
    page=page.replace('function render(index) {','function render(index) {\n  renderBusiness(index);',1)
    page=page.replace('const requested=Number(',_SCRIPT+'\nconst requested=Number(',1)
    return page


def render_comparison_html(result):
    def esc(value):return html.escape(str(value))
    def fmt(value):return '—' if value is None else f'{value:.2f}'
    rows=[]
    mapping={'workflow_rule':'rule','workflow_pets':'pets','workflow_structured':'structured'}
    for name,metrics in result['methods'].items():
        accuracy=[]
        for horizon in ('1','3','5'):
            values=result['independent_factual_prediction']['horizons'][horizon]['dynamic'].get(mapping.get(name),{})
            correct,total=values.get('joint_correct',0),values.get('windows',0)
            accuracy.append(f'{100*correct/total:.1f}% ({correct}/{total})' if total else '不适用')
        motion=metrics['motion_safety'];bad=sum(motion[k] for k in ('vertex_collisions','edge_swaps','invalid_edges'))
        rows.append('<tr>'+''.join('<td>'+v+'</td>' for v in [
            '<a href="'+esc(name)+'/report.html">'+esc(name)+'</a>',
            esc(f"{metrics['jobs_completed']}/{metrics['jobs']} ({100*metrics['completion_rate']:.1f}%)"),
            fmt(metrics['mean_flow_completed'])+' / '+fmt(metrics['restricted_mean_flow']),
            fmt(metrics['mean_stage_wait_per_job']),esc(metrics['resource']['capacity_wait_ticks']),
            esc(metrics['station_unavailable_wait_ticks']),esc(len(metrics['resource']['violations'])),esc(bad),
            fmt(metrics['decision_wall_mean_ms'])+' / '+fmt(metrics['decision_wall_p95_ms']),
            esc(' / '.join(accuracy))])+'</tr>')
    coverage=[]
    for horizon in ('1','3','5'):
        c=result['independent_factual_prediction']['horizons'][horizon]['dynamic_coverage']
        coverage.append(f"{horizon} 步：选中 {c.get('selected',0)} / 合格 {c.get('eligible',0)} / 候选 {c.get('candidate_windows',0)}")
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>样本转运策略对照</title><style>body{font-family:system-ui,sans-serif;max-width:1400px;margin:auto;padding:24px;color:#17283b;background:#f3f6fa}section{background:white;padding:20px;border-radius:12px;overflow:auto}table{border-collapse:collapse;width:100%;min-width:1100px}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left}a{color:#1769aa}p{line-height:1.6}</style></head><body><h1>样本转运策略对照</h1><section><p>合成任务和公开地图；不代表真实实验室性能。相同场景、任务和外生扰动，未重新训练模型。</p><table><thead><tr><th>方案及回放</th><th>业务任务完成</th><th>平均 / 受限流转（步）</th><th>阶段等待（步/任务）</th><th>容量等待（步）</th><th>设备不可用等待（步）</th><th>容量违规</th><th>离散安全违规</th><th>决策平均 / P95（ms）</th><th>1 / 3 / 5 步联合准确率</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table><p>预测准确率由同一条最近优先事实轨迹离线评价，三种模型使用相同窗口；不是各自闭环轨迹的混合分母。最近优先不调用世界模型。</p><p>窗口覆盖：'''+esc('；'.join(coverage))+'''。窗口可能重叠；此场景已用于 PR #15，属于系统演示，不是新的独立确认。</p><p>决策墙钟包含真实预测轨迹记录的开销；与未记录轨迹的历史实验耗时不直接等同。</p></section></body></html>'''
