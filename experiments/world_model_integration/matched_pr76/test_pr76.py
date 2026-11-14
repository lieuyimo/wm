"""PR76 factual action/device-match test suite; never synthetic score assertions."""
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from experiments.world_model_integration.matched_pr76.pr76_factual_contract import (
    ARMS, HORIZONS, PARTICLE_SEEDS, SOURCE_PR74_CHECKPOINT_SHA,
    ExistingTrainPublicHazard, strict_state_error,
)
from experiments.world_model_integration.matched_pr76.pr76_eval import (
    summarize_conditioned,
)


@dataclass(frozen=True)
class FakeSample:
    id: str
    status: str

@dataclass(frozen=True)
class FakeRobot:
    id: str
    position: tuple
    phase: str

@dataclass(frozen=True)
class FakeStation:
    id: str
    service_in_progress: bool

def fake_state(tick, samples):
    return SimpleNamespace(tick=tick,
            samples=tuple(FakeSample(*x) for x in samples),
            robots=(FakeRobot("R1",(0,0),"idle"),),
            stations=(FakeStation("A",False),))


class PR76NoTruthLeakageAndEvaluationControls(unittest.TestCase):
    def test_symmetric_union_state_score_penalizes_missing_and_early_stage2(self):
        truth=fake_state(5,[("J1::stage1","completed"),("J1::stage2","waiting")])
        prediction=fake_state(5,[("J1::stage1","completed"),("J2::stage2","waiting")])
        value=strict_state_error(prediction,truth)
        self.assertEqual(value["visible_task_union"],3)
        self.assertEqual(value["task_status_errors"],2)
        self.assertEqual(value["missing_factual_tasks"],1)
        self.assertEqual(value["premature_extra_tasks"],1)
        self.assertAlmostEqual(value["task_status_error_rate"],2/3)
        self.assertEqual(value["completed_abs_error"],0)

    def test_timing_mismatch_rejected(self):
        with self.assertRaisesRegex(AssertionError,"tick mismatch"):
            strict_state_error(fake_state(6,[]),fake_state(5,[]))

    def test_original_public_discrete_hazard_only_train_and_same_phase(self):
        train=[{"station":"A","phase":"to_pickup","active_elapsed":2,"event":i}
               for i in (0,1,1)]
        train+= [{"station":"A","phase":"to_dropoff","active_elapsed":1,"event":i}
                 for i in (0,1,0)]
        control=ExistingTrainPublicHazard(train)
        a=control.query_probability(None,{"station_id":"A","phase":"to_pickup"},2,3.)
        b=control.query_probability(None,{"station_id":"A","phase":"to_dropoff"},1,3.)
        self.assertAlmostEqual(a,3/5)
        self.assertAlmostEqual(b,2/5)
        self.assertEqual(control.calls,2)
        with self.assertRaisesRegex(ValueError,"missing"):
            control.query_probability(None,{"station_id":"A","phase":"to_pickup"},None,3.)

    def test_strict_all_arm_denominators_and_no_false_full_horizon_ci(self):
        records=[]
        for j in range(12):
            for seed in PARTICLE_SEEDS:
                for arm in ARMS:
                    full=not (j==0 and seed==PARTICLE_SEEDS[0] and arm=="existing_deepsets")
                    steps=[{"task_status_error_rate":0.1,"completed_abs_error":0}]
                    rows=steps*5 if full else steps*2
                    records.append({
                        "scene":f"S{j}","anchor_name":"initial_t0",
                        "horizon":5,"seed":seed,"arm":arm,
                        "full_horizon":full,"legal_steps":len(rows),
                        "failure_kind":None if full else "factual_action_not_executable_in_predicted_world",
                        "offline_model_service_queries":0,
                        "errors":rows,
                    })
        result=summarize_conditioned(records,5,"initial_t0")
        self.assertEqual(result["attempted_matched_scene_seed_triples"],36)
        self.assertEqual(result["fully_action_device_matched_triples"],35)
        self.assertEqual(result["arm"]["existing_deepsets"]["full_horizon"],35)
        self.assertFalse(result["full_horizon_matched_terminal"]["all_pairs_usable"])
        self.assertFalse(result["full_horizon_matched_terminal"]["scene_bootstrap"]["available"])

    def test_only_frozen_scoped_protocol_and_training_weights(self):
        self.assertEqual(HORIZONS,(5,20))
        self.assertEqual(len(PARTICLE_SEEDS),3)
        self.assertEqual(ARMS,("rule","existing_deepsets","train_public_hazard"))
        self.assertEqual(SOURCE_PR74_CHECKPOINT_SHA,
            "a08b257c7f8b8a6191b954e9adb117ef8076f3ff93d5373c347d4735e1e45f02")


if __name__=="__main__":
    unittest.main()
