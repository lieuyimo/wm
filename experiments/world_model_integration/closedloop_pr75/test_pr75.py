"""PR75 tests are structural only, never simulated trained performance."""
import unittest
from dataclasses import dataclass

from experiments.world_model_integration.closedloop_pr75.pr75_legal_h20 import (
    ARMS, HORIZONS, SEEDS, summarize, scene_comparison, EXPECTED_CHECKPOINT_SHA256,
)


@dataclass(frozen=True)
class Sample:
    status: str
    stage_index: int


@dataclass(frozen=True)
class Station:
    service_in_progress: bool
    resource_occupied: int


@dataclass(frozen=True)
class Obs:
    samples: tuple
    stations: tuple


class PR75ExistingModelContract(unittest.TestCase):
    def test_terminal_summary_distinguishes_task_lifecycle(self):
        obs=Obs(samples=(
            Sample("completed",0),
            Sample("waiting",0),
            Sample("in_transit",1)),
            stations=(Station(True,1), Station(False,0)))
        value=summarize(obs)
        self.assertEqual(value["completed_stages"],1)
        self.assertEqual(value["visible_stage1"],2)
        self.assertEqual(value["visible_stage2"],1)
        self.assertEqual(value["active_services"],1)
        self.assertEqual(value["busy_ports"],1)

    def test_only_actual_original_PR74_checkpoint_admitted(self):
        self.assertEqual(
            EXPECTED_CHECKPOINT_SHA256,
            "a08b257c7f8b8a6191b954e9adb117ef8076f3ff93d5373c347d4735e1e45f02")
        self.assertEqual(ARMS, ("rule","existing_deepsets"))
        self.assertEqual(HORIZONS,(5,20))
        self.assertEqual(SEEDS,(1810,1811,1812))

    def test_no_unbalanced_arm_claimed_as_paired(self):
        runs=[]
        for i in range(12):
            for seed in SEEDS:
                runs.append({
                    "scene":f"scene{i}","seed":seed,"horizon":20,
                    "arm":"rule", "new_stage1_arrivals":[],
                    "external_device_changes":0,
                    "terminal":{"completed_stages":0}})
        with self.assertRaisesRegex(AssertionError,"common source"):
            scene_comparison(runs,20)


if __name__=="__main__":
    unittest.main()
