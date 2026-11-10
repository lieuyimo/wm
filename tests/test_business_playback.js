// Exercise the actual offline business report without a frontend framework.
const assert=require('assert'),fs=require('fs'),vm=require('vm'),path=require('path');
const root=process.argv[2];assert(root,'pass the demo output directory');
const run=JSON.parse(fs.readFileSync(path.join(root,'run.json'),'utf8'));
const html=fs.readFileSync(path.join(root,'report.html'),'utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/);assert(script,'missing playback script');
class Element{
  constructor(){this.children=[];this.listeners={};this.className='';this.textContent='';this.value='0';this.style={};this.attributes={};
    this.classList={add:name=>{this.className+=' '+name}}}
  appendChild(child){this.children.push(child)}
  replaceChildren(){this.children=[]}
  addEventListener(name,callback){this.listeners[name]=callback}
  setAttribute(name,value){this.attributes[name]=String(value)}
}
const elements={};let interval;
const document={getElementById:id=>elements[id]||(elements[id]=new Element()),
  createElement:()=>new Element(),createElementNS:()=>new Element()};
vm.runInNewContext(script[1],{document,window:{location:{search:'?step=0'}},URLSearchParams,
  setInterval:callback=>{interval=callback;return 1},clearInterval:()=>{}},{timeout:20000});
const slider=elements.tick;
const text=element=>[element.textContent,...element.children.map(text)].join(' ');
const setTick=tick=>{const i=run.history.findIndex(frame=>frame.tick===tick);assert(i>=0);slider.value=String(i);slider.listeners.input()};
assert(text(elements['business-summary']).includes(`${run.business.summary.jobs_completed}/${run.business.summary.jobs}`));
assert.equal(elements['business-robots'].children.length,run.history[0].robots.length);
assert.equal(elements['business-stations'].children.length,run.history[0].stations.length);
for(let i=0;i<run.history[0].stations.length;i++){
  const station=run.history[0].stations[i];
  assert(text(elements['business-stations'].children[i]).includes(`${station.resource_occupied}/${station.resource_capacity}`));
}
const middle=run.history.find(frame=>frame.samples.some(s=>s.stage_index===0&&s.status==='completed'&&
  frame.samples.some(next=>next.job_id===s.job_id&&next.stage_index===1&&next.status!=='completed')));
if(middle){setTick(middle.tick);assert(text(elements['business-jobs']).includes('业务任务进行中'))}
const decision=run.decisions.find(d=>d.plans.length>1&&d.plans.some(p=>p.prediction_particles?.length));
if(decision){
  setTick(decision.tick);
  const candidate=elements['forecast-candidate'],forecast=elements['business-forecast'];
  assert(elements['business-decision'].textContent.includes(JSON.stringify(decision.legal_candidates)));
  assert(forecast.textContent.includes('【预测 t='));
  assert(forecast.textContent.includes('【事实 t='));
  const next=(decision.selected_index+1)%decision.plans.length;candidate.value=String(next);candidate.listeners.change();
  assert(forecast.textContent.includes('该候选未执行，没有其反事实观测'));
  assert(!forecast.textContent.includes('【事实 t='));
  assert(forecast.textContent.includes(JSON.stringify(decision.plans[next].score)));
}else if(run.business.model_name==='none'){
  assert(elements['business-forecast'].textContent.includes('不调用世界模型'));
}
const recovery=run.events.find(e=>e.type==='lacam3_recovery'&&e.triggered);
if(recovery){setTick(recovery.tick);assert(text(elements['business-events']).includes('LaCAM3 恢复'))}
slider.value=String(slider.max);slider.listeners.input();
assert(text(elements['business-summary']).includes(`${run.business.summary.jobs_completed}/${run.business.summary.jobs}`));
slider.value='0';slider.listeners.input();elements.play.listeners.click();interval();
assert(elements.time.textContent.includes('第 1 步'));
console.log('Business playback verified:',run.policy,'frames',run.history.length,'jobs',run.business.summary.jobs_completed+'/'+run.business.summary.jobs);
