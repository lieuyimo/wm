"""PR62 read-only evidence and checkpoint-dependent phase arithmetic tests."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.world_model_integration.audit_pr62.pr62_readonly import audit
from experiments.world_model_integration.audit_pr62.pr62_device_events import (
    score_group,station_stage
)


class ReadonlyPR61EvidenceTests(unittest.TestCase):
    def test_public_PR61_72_original_validation_rows_and_12_independent_scenes(self):
        root=Path("examples/evidence/pr60-gpu-training-20261112")
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/"readonly-audit.json"
            result=audit(root/"per_scenario.csv",root/"summary.json",out)
            self.assertEqual(result["n_actual_independent_validation_scenes"],12)
            self.assertEqual(result["n_distinct_validation_maps"],2)
            self.assertEqual(result["n_seed_by_scenario_paired_observations_not_independent"],36)
            self.assertEqual(result["public_state_next_tick_MSE"]["num_map_on_better"],11)
            self.assertEqual(result["station_device_availability_Brier"]["num_map_on_better"],7)
            self.assertTrue(result["per_map"]["0"]["device_brier"]["mean_on_minus_off"]<0)
            self.assertTrue(result["per_map"]["1"]["device_brier"]["mean_on_minus_off"]>0)
            self.assertEqual(result["CUDA_or_model_calls"],0)
            self.assertEqual(result["blind_test_files_read"],0)
            self.assertTrue(out.exists())
            with self.assertRaises(FileExistsError):
                audit(root/"per_scenario.csv",root/"summary.json",out)

    def test_station_true_event_types_not_treated_as_available_predictive_inputs(self):
        self.assertEqual(station_stage(1.,1.),"healthy_hold")
        self.assertEqual(station_stage(1.,0.),"fault_onset")
        self.assertEqual(station_stage(0.,0.),"continuing_down")
        self.assertEqual(station_stage(0.,1.),"recovery")

    def test_persistence_error_and_model_brier_math(self):
        # Input: (model prob P(next available), TRUE next available, CURRENT available)
        x=score_group([(0.8,1.,1.),(0.2,0.,1.)])
        self.assertEqual(x["n"],2)
        self.assertAlmostEqual(x["model_brier"],0.04)
        self.assertAlmostEqual(x["trivial_previous_device_state_brier"],0.5)
        self.assertAlmostEqual(x["model_minus_persistence_brier"],-0.46)
        self.assertEqual(score_group([]),{"n":0,"available":False})


if __name__=="__main__":
    unittest.main()
