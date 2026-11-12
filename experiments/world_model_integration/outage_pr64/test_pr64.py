"""PR64 immutable CPU randomized private outage time and heldout gate tests."""
import tempfile
import unittest
from pathlib import Path

from experiments.world_model_integration.outage_pr64.pr64_source import (
    read_protocol,private_outage_shift,onset_start,run
)
from experiments.world_model_integration.outage_pr64.pr64_gate import (
    clock4,exact_age4,fit,risk_set_score
)
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL
from experiments.world_model_integration.topology_pr56.pr56_public_map import build_topology_business
import json


class PrecommittedOutageShiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p,cls.old=read_protocol()
        cls.stress=json.loads(PR48_PROTOCOL.read_text())

    def test_no_original_BLIND_or_neural_GPU_training(self):
        self.assertEqual(len(self.p["cohort"]),36)
        self.assertEqual({s:sum(c["split"]==s for c in self.p["cohort"])
                          for s in ("train","validation")},{"train":24,"validation":12})
        self.assertEqual(self.p["splits"]["BLIND_TEST"]["count"],0)
        self.assertEqual(self.p["resource"]["GPU"],0)
        self.assertEqual(self.p["resource"]["new_model_training"],0)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(PermissionError):
                run("test",Path(d)/"forbidden-test-data.json")

    def test_deterministic_SHA_random_start_has_no_TRAIN_VAL_overlap(self):
        all_obs={}
        for split in ("train","validation"):
            allowed=self.p["splits"][split]["allowed_private_start_ticks"]
            values=set()
            for c in self.p["cohort"]:
                if c["split"]!=split:continue
                starts=[]
                for ordinal,sid in enumerate([c["station_id"]]+(
                      [c["secondary_station_id"]] if c["regime"]=="multi_fault" else [])):
                    t=onset_start(c["seed"],c["regime"],sid,ordinal,allowed)
                    if t in starts:
                        t=allowed[(allowed.index(t)+1)%len(allowed)]
                    starts.append(t)
                    values.add(t)
            all_obs[split]=values
            self.assertTrue(values<=set(allowed))
        self.assertFalse(all_obs["train"]&all_obs["validation"])

    def test_real_original_world_only_hidden_start_changes_not_durations_or_physics(self):
        for split in ("train","validation"):
            case=next(c for c in self.p["cohort"]
                      if c["split"]==split and c["regime"]=="multi_fault")
            world,geo=build_topology_business(case,self.stress,self.old)
            changed,starts,old_durations=private_outage_shift(
                world,case,self.p["splits"][split]["allowed_private_start_ticks"])
            self.assertEqual(changed.rules,world.rules)
            self.assertEqual(changed.jobs,world.jobs)
            self.assertEqual(changed.scenario.grid,world.scenario.grid)
            self.assertEqual(changed.scenario.robots,world.scenario.robots)
            self.assertEqual(len(starts),2)
            self.assertNotEqual(starts[0],starts[1])
            self.assertTrue(all(t in self.p["splits"][split]["allowed_private_start_ticks"]
                                for t in starts))
            fault={case["station_id"],case["secondary_station_id"]}
            for sid in world.scenario.stations:
                old=world.scenario.stations[sid]
                new=changed.scenario.stations[sid]
                if sid in fault:
                    self.assertEqual(
                        old.outages[0][1]-old.outages[0][0],
                        new.outages[0][1]-new.outages[0][0])
                    self.assertEqual(old.service_ticks,new.service_ticks)
                else:
                    self.assertEqual(old,new)

    def test_uniform_four_tick_time_buckets_do_not_single_out_PR63_tick_6_12(self):
        self.assertEqual(clock4({"tick":6}),"1")
        self.assertEqual(clock4({"tick":7}),"1")
        self.assertEqual(clock4({"tick":12}),"3")
        self.assertEqual(clock4({"tick":13}),"3")
        self.assertEqual(exact_age4({"age_exact":1}),"0")
        self.assertEqual(exact_age4({"age_exact":8}),"1")

    def test_hazard_fit_using_only_train_risk_set_no_parameter_updates(self):
        def r(scene,split,t,cur,y,age):
            phase=("healthy_hold" if cur and y else
                   "fault_onset" if cur and not y else
                   "recovery" if not cur and y else "continuing_down")
            return {"scene":scene,"split":split,"tick":t,"age_h4":min(age,4),
                    "age_exact":age,"risk_set":"up" if cur else "down",
                    "current":cur,"target":y,"phase":phase,"map":"0",
                    "regime":"congestion"}
        train=[r("train","train",i,1,1 if i!=3 else 0,i+1) for i in range(6)]
        train += [r("train","train",i,0,1 if i==4 else 0,i-1) for i in (3,4,5)]
        val=[r("dev","validation",i,1,1 if i!=3 else 0,i+1) for i in range(6)]
        val +=[r("dev","validation",i,0,1 if i==4 else 0,i-1) for i in (3,4,5)]
        params=fit(train)
        self.assertTrue(params["fit_only_on_train"])
        self.assertEqual(params["constant"]["up"]["events"],1)
        self.assertEqual(params["constant"]["down"]["events"],1)
        metrics=risk_set_score(val,params)
        self.assertEqual(set(metrics),{"constant","age_h4","clock4","exact_age4"})
        self.assertTrue(all(0<=m["overall"]["brier"]<=1 for m in metrics.values()))


if __name__=="__main__":
    unittest.main()
