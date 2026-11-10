"""Check actual rendered Chromium DOM against the same run.json data."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess


class TextById(HTMLParser):
    def __init__(self):
        super().__init__();self.stack=[];self.values={};self.in_script=False

    def handle_starttag(self,tag,attrs):
        identifier=dict(attrs).get('id')
        if tag=='script':self.in_script=True
        if identifier:self.values.setdefault(identifier,[])
        if tag not in {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}:
            self.stack.append((tag,identifier))

    def handle_endtag(self,tag):
        if tag=='script':self.in_script=False
        for index in range(len(self.stack)-1,-1,-1):
            if self.stack[index][0]==tag:
                del self.stack[index:];break

    def handle_data(self,data):
        if not self.in_script:
            for _,identifier in self.stack:
                if identifier:self.values[identifier].append(data)

    def text(self,key):return ' '.join(self.values.get(key,[]))


def run(root,chrome,focus,save_dom=None):
    root=Path(root).resolve();data=json.loads((root/'run.json').read_text())
    if focus=='recovery':
        event=next(e for e in data['events'] if e['type']=='lacam3_recovery' and e['triggered'])
        index=next(i for i,f in enumerate(data['history']) if f['tick']==event['tick'])
    elif focus=='end':
        index=len(data['history'])-1
    else:
        decision=next((d for d in data['decisions'] if d['plans']),None)
        index=next((i for i,f in enumerate(data['history']) if decision and f['tick']==decision['tick']),0)
    url=(root/'report.html').as_uri()+f'?step={index}'
    proc=subprocess.run([chrome,'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
                         '--no-first-run','--dump-dom',url],capture_output=True,text=True,timeout=45,check=True)
    if save_dom:Path(save_dom).write_text(proc.stdout,encoding='utf-8')
    page=TextById();page.feed(proc.stdout);frame=data['history'][index]
    assert f"{data['business']['summary']['jobs_completed']}/{data['business']['summary']['jobs']}" in page.text('business-summary')
    for robot in frame['robots']:
        assert robot['id'] in page.text('business-robots')
        assert f"({robot['position'][0]}, {robot['position'][1]})" in page.text('business-robots')
    for station in frame['stations']:
        assert station['id'] in page.text('business-stations')
        assert f"{station['resource_occupied']}/{station['resource_capacity']}" in page.text('business-stations')
    if focus=='recovery':assert 'LaCAM3 恢复' in page.text('business-events')
    elif data['business']['model_name']=='none':assert '不调用世界模型' in page.text('business-forecast')
    else:
        assert '【预测 t=' in page.text('business-forecast')
        assert '【事实 t=' in page.text('business-forecast')
        assert data['business']['model_name'] in page.text('business-decision')
    print(f"Browser DOM verified: {data['policy']} tick={frame['tick']} jobs="
          f"{data['business']['summary']['jobs_completed']}/{data['business']['summary']['jobs']}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--chrome',default=shutil.which('google-chrome') or shutil.which('chromium'))
    parser.add_argument('--focus',choices=('decision','recovery','end'),default='decision')
    parser.add_argument('--save-dom',type=Path)
    args=parser.parse_args()
    if not args.chrome:parser.error('Chrome/Chromium 未安装')
    run(args.root,args.chrome,args.focus,args.save_dom)


if __name__=='__main__':main()
