"""PR71 CPU-only semantic safeguards. No trained weight access."""
import unittest
from dataclasses import dataclass

from experiments.world_model_integration.audit_pr71.pr71_phase_audit import (
    per_scene_transitions,
)
from experiments.world_model_integration.audit_pr71.pr71_public_arrivals import (
    apply_current_public_stage1,
)


def row(t, current, future, mask=None):
    onehot = {
        "waiting": [1., 0., 0., 0., 0., 0.],
        "assigned": [0., 1., 0., 0., 1., 0.],
        "in_transit": [0., 0., 1., 0., 1., 0.],
        "completed": [0., 0., 0., 1., 1., 0.],
    }
    return {
        "input": {"public_tick": t, "graph": {"nodes": [
            {"kind": "task", "status": current, "id": "task:0"},
            {"kind": "station", "id": "station:0"},
        ]}},
        "target": {"next_public_tick": t + 1,
                   "node_target": [onehot[future], [1., 1., 0., 0., 0., 0.]],
                   "node_mask": [mask or [1.] * 6, [1.] * 6]}
    }


@dataclass(frozen=True)
class Sample:
    id: str
    stage_index: int
    job_id: str
    status: str = "waiting"
    robot_id: str | None = None
    release_tick: int = 0


@dataclass(frozen=True)
class Obs:
    tick: int
    samples: tuple


class PR71PurePublicContractTests(unittest.TestCase):
    def test_pickup_to_in_transit_is_not_final_delivery(self):
        case = {"id": "case", "raw_actual_public_world_transitions": [
            row(2, "assigned", "in_transit"),
            row(8, "in_transit", "completed")
        ]}
        found = per_scene_transitions(case)
        self.assertEqual(found["first_observed_tick"]["pickup_to_in_transit"], 3)
        self.assertEqual(found["first_observed_tick"]["delivery_to_completed"], 9)
        self.assertEqual(found["events"], {
            "pickup_to_in_transit": 1, "delivery_to_completed": 1
        })

    def test_only_observed_and_masked_task_transitions_count(self):
        case = {"id": "case", "raw_actual_public_world_transitions": [
            row(1, "assigned", "in_transit", [0.] * 6),
            row(2, "waiting", "assigned"),
            row(3, "in_transit", "completed"),
        ]}
        found = per_scene_transitions(case)
        self.assertIsNone(found["first_observed_tick"]["pickup_to_in_transit"])
        self.assertEqual(found["events"]["delivery_to_completed"], 1)

    def test_clock_mismatch_rejected(self):
        r = row(0, "assigned", "in_transit")
        r["target"]["next_public_tick"] = 7
        with self.assertRaisesRegex(ValueError, "chronological"):
            per_scene_transitions({"id": "bad", "raw_actual_public_world_transitions": [r]})

    def test_stage1_cannot_be_injected_from_future_tick(self):
        current = Obs(4, ())
        future = Obs(5, (Sample("J7::stage1", 0, "J7", release_tick=5),))
        with self.assertRaisesRegex(ValueError, "CURRENT tick"):
            apply_current_public_stage1(current, future, None, None)

    def test_already_visible_stage1_missing_from_prediction_is_not_silently_repaired(self):
        before = Obs(2, (Sample("J7::stage1", 0, "J7", release_tick=2),))
        actual = Obs(3, (Sample("J7::stage1", 0, "J7", release_tick=2),))
        predicted = Obs(3, ())
        with self.assertRaisesRegex(ValueError, "already public"):
            apply_current_public_stage1(predicted, actual, before, None)

    def test_factual_stage2_is_never_injected(self):
        current = Obs(0, (Sample("J5::stage1", 0, "J5"),))
        actual = Obs(0, (Sample("J5::stage1", 0, "J5"),
                         Sample("J5::stage2", 1, "J5")))
        result, added = apply_current_public_stage1(current, actual, None, None)
        self.assertEqual(added, ())
        self.assertIs(result, current)


if __name__ == "__main__":
    unittest.main()
