from contextlib import redirect_stderr,redirect_stdout
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from business_transport import demo
from business_transport.dynamics import WorkflowDynamics

PIBT=os.environ.get('WM_PIBT_LIBRARY');LACAM=os.environ.get('WM_LACAM3_WORKER');SOURCES=os.environ.get('WM_HOSPITAL_SOURCES')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


class InvokeMixin:
    def invoke(self,args):
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):return demo.main(args)


class DemoTests(InvokeMixin,unittest.TestCase):
    def test_three_stage_capture_is_the_actual_online_rollout_and_does_not_overwrite(self):
        captured=[];original=WorkflowDynamics.predict_next
        def spy(model,observation,assignments):
            result=original(model,observation,assignments)
            captured.append({'tick':result.tick,'actions':[asdict(a) for a in assignments],
                'robots':[asdict(r) for r in result.robots],'samples':[asdict(s) for s in result.samples],
                'stations':[asdict(s) for s in result.stations]})
            return result
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            with patch.object(WorkflowDynamics,'predict_next',spy):
                self.invoke(['run','--scenario','three_stage','--motion','legacy','--policy','workflow_rule','--output',str(root)])
            data=read(root/'run.json');summary=read(root/'summary.json');decisions=read(root/'decisions.json')
            events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
            self.assertEqual(summary,data['business']['summary']);self.assertEqual(decisions['decisions'],data['decisions'])
            self.assertEqual(events,data['events']);self.assertEqual(summary['jobs_completed'],1)
            self.assertEqual(data['summary']['completed_count'],3)
            self.assertEqual([e['sample_id'] for e in events if e['type']=='stage_ready'],['J1::stage2','J1::stage3'])
            stored=[step for decision in data['decisions'] for plan in decision['plans']
                    for particle in plan['prediction_particles'] for step in particle['steps']]
            self.assertEqual(len(stored),len(captured));self.assertGreater(len(captured),0)
            for record,actual in zip(stored,captured):
                for key in actual:self.assertEqual(record[key],json.loads(json.dumps(actual[key])))
            self.assertNotIn('service_remaining',json.dumps(data['history']))
            self.assertNotIn('service_remaining',json.dumps(stored))
            self.assertTrue((root/'report.html').is_file())
            previous=(root/'run.json').read_bytes()
            with self.assertRaises(SystemExit):
                self.invoke(['run','--scenario','three_stage','--motion','legacy','--policy','workflow_rule','--output',str(root)])
            self.assertEqual((root/'run.json').read_bytes(),previous)
            subprocess.run(['node','tests/test_business_playback.js',str(root)],check=True,capture_output=True,text=True)

    def test_missing_dependencies_and_invalid_model_are_explicit(self):
        with self.assertRaisesRegex(FileNotFoundError,'PIBT'):demo.require_native('hybrid',None,None)
        with self.assertRaisesRegex(ValueError,'不能静默替换'):demo.resolve_model('workflow_rule','pets')
        with patch.object(demo,'cp_model',None):
            with self.assertRaisesRegex(RuntimeError,'OR-Tools'):demo.model_for('rule',None,None,'legacy',None,None,1)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(demo,'CHECKPOINT',Path(tmp)/'missing.pt'):
                with self.assertRaisesRegex(FileNotFoundError,'PETS'):demo.model_for('pets',None,None,'legacy',None,None,1)

    def test_nearest_does_not_call_or_label_a_world_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'nearest'
            self.invoke(['run','--scenario','three_stage','--motion','legacy','--policy','nearest','--model','none','--output',str(root)])
            data=read(root/'run.json')
            self.assertEqual(data['business']['model_name'],'none')
            self.assertEqual(data['business']['summary']['prediction_calls'],0)
            self.assertTrue(all(not d['plans'] for d in data['decisions']))
            subprocess.run(['node','tests/test_business_playback.js',str(root)],check=True,capture_output=True,text=True)


@unittest.skipUnless(PIBT and LACAM and Path(PIBT).is_file() and Path(LACAM).is_file(),
                     'set WM_PIBT_LIBRARY and WM_LACAM3_WORKER')
class NativeDemoTests(InvokeMixin,unittest.TestCase):
    def test_pibt_backend_executes_the_existing_native_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'pibt'
            self.invoke(['run','--scenario','three_stage','--motion','pibt','--policy','workflow_rule',
                         '--pibt-library',PIBT,'--output',str(root)])
            data=read(root/'run.json');self.assertEqual(data['business']['motion_backend'],'pibt')
            self.assertGreater(data['business']['summary']['pibt_calls'],0)
            self.assertTrue(any(e['type']=='pibt_step' for e in data['events']))
            self.assertEqual(data['business']['summary']['model_fallbacks'],0)

    def test_fixed_comparison_uses_one_factual_reference_for_all_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'comparison'
            self.invoke(['compare','--scenario','three_stage','--motion','hybrid',
                         '--pibt-library',PIBT,'--lacam-worker',LACAM,'--output',str(root)])
            data=read(root/'comparison.json')
            self.assertEqual(data['reference_policy'],'nearest')
            self.assertEqual(set(data['methods']),set(demo.POLICIES))
            self.assertGreater(data['methods']['workflow_pets']['pets_forward_calls'],0)
            self.assertGreater(data['methods']['workflow_structured']['structured_service_calls'],0)
            self.assertEqual(data['methods']['nearest']['prediction_calls'],0)
            for horizon in ('1','3','5'):
                stats=data['independent_factual_prediction']['horizons'][horizon]['dynamic']
                self.assertEqual(len({stats[name].get('windows',0) for name in ('rule','pets','structured')}),1)
            for policy in demo.POLICIES:
                self.assertTrue((root/policy/'report.html').is_file())
                self.assertEqual(read(root/policy/'run.json')['business']['model_name'],demo.POLICIES[policy] or 'none')
                subprocess.run(['node','tests/test_business_playback.js',str(root/policy)],check=True,capture_output=True,text=True)
            page=(root/'comparison.html').read_text();self.assertIn('同一条最近优先事实轨迹',page)

    def test_outage_and_capacity_wait_are_observed_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'outage'
            self.invoke(['run','--scenario','empty8','--regime','workflow_outage','--seed','71003',
                         '--policy','nearest','--motion','hybrid','--pibt-library',PIBT,'--lacam-worker',LACAM,
                         '--output',str(root)])
            data=read(root/'run.json');events=data['events'];summary=data['business']['summary']
            self.assertGreater(summary['resource']['capacity_wait_ticks'],0)
            self.assertGreater(summary['station_unavailable_wait_ticks'],0)
            self.assertTrue(any(e['type']=='resource_wait' for e in events))
            self.assertTrue(any(e['type']=='service_paused_observed' for e in events))
            self.assertTrue(any(e['type']=='device_status_observed' and not e['available'] for e in events))
            self.assertFalse(summary['resource']['violations'])


@unittest.skipUnless(PIBT and LACAM and SOURCES and Path(PIBT).is_file() and Path(LACAM).is_file(),
                     'set both native paths and WM_HOSPITAL_SOURCES')
class ClinicDemoTests(InvokeMixin,unittest.TestCase):
    def test_frozen_clinic_recovery_is_recorded_and_rendered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'clinic'
            self.invoke(['run','--scenario','clinic','--seed','61000','--policy','nearest','--motion','hybrid',
                         '--pibt-library',PIBT,'--lacam-worker',LACAM,'--source-dir',SOURCES,'--output',str(root)])
            data=read(root/'run.json');s=data['business']['summary']
            self.assertEqual((s['jobs_completed'],s['jobs'],data['summary']['ticks']),(5,5,115))
            self.assertEqual((s['lacam3_triggers'],s['lacam3_successes'],s['lacam3_cache_steps']),(2,2,11))
            self.assertEqual(len([e for e in data['events'] if e['type']=='lacam3_recovery' and e['triggered']]),2)
            subprocess.run(['node','tests/test_business_playback.js',str(root)],check=True,capture_output=True,text=True)


if __name__=='__main__':unittest.main()
