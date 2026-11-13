"""PR72 source-only tests. Full original PR64 scenes exercised by CPU workflow."""
import unittest

from experiments.world_model_integration.identifiability_pr72.pr72_risk_and_alias import (
    PHASE_TARGETS, raw_head_targets,
)
from experiments.world_model_integration.decision_learning_pr51.pr51_train import (
    node_features,
)


class PR72InformationContractTests(unittest.TestCase):
    def test_true_next_task_class_labels_distinguish_service_phases(self):
        one_hot = {
            "assigned": [0, 1, 0, 0, 1, 0],
            "in_transit": [0, 0, 1, 0, 1, 0],
            "completed": [0, 0, 0, 1, 1, 0],
        }
        rows = []
        for current, future in (("assigned", "in_transit"),
                                ("in_transit", "completed"),
                                ("completed", "completed")):
            rows.append({
                "input": {"graph": {"nodes": [
                    {"id": "task:0", "kind": "task", "status": current}
                ]}},
                "target": {
                    "node_target": [one_hot[future]],
                    "node_mask": [[1] * 6],
                },
            })
        result = raw_head_targets({"raw_actual_public_world_transitions": rows})
        self.assertEqual(result["task_node_labels"], 3)
        self.assertEqual(result["transitions"]["assigned->in_transit"], 1)
        self.assertEqual(result["transitions"]["in_transit->completed"], 1)
        self.assertEqual(result["transitions"]["completed->completed"], 1)

    def test_target_labels_no_fabricated_zero_mask(self):
        data = {
            "raw_actual_public_world_transitions": [{
                "input": {"graph": {"nodes": [{
                    "id": "task:0", "kind": "task", "status": "assigned"
                }]}},
                "target": {
                    "node_target": [[0, 0, 1, 0, 1, 0]],
                    "node_mask": [[0, 0, 0, 0, 1, 1]],
                }
            }]
        }
        self.assertEqual(raw_head_targets(data)["task_node_labels"], 0)

    def test_original_graph_omits_rule_visible_effective_progress(self):
        robot = {
            "kind": "robot", "id": "robot:0", "pos": [2, 1],
            "phase": "to_pickup", "busy": True, "service_elapsed": 4,
        }
        station = {
            "kind": "station", "id": "station:0", "pos": [2, 1],
            "device_available": True, "resource_occupied": 1,
            "waiting_pickups": 0, "approaching_pickups": 0,
        }
        x = node_features(robot, set())
        s = node_features(station, set())
        robot_more_public = dict(robot, service_active_elapsed=1, service_paused_elapsed=3)
        station_more_public = dict(station, observed_effective_service_mean=10.0)
        self.assertEqual(x, node_features(robot_more_public, set()))
        self.assertEqual(s, node_features(station_more_public, set()))
        self.assertAlmostEqual(x[9], 0.2)

    def test_phase_outcomes_not_mixed(self):
        self.assertEqual(PHASE_TARGETS, {
            "to_pickup": "in_transit", "to_dropoff": "completed",
        })


if __name__ == "__main__":
    unittest.main()
