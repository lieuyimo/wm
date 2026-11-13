"""Pure checks for PR70's public-arrival adapter."""
from __future__ import annotations

from dataclasses import dataclass
import unittest

from experiments.world_model_integration.lifecycle_pr70.pr70_lifecycle_eval import (
    inject_public_stage1,
)


@dataclass(frozen=True)
class Sample:
    id: str
    status: str = "waiting"


@dataclass(frozen=True)
class Observation:
    samples: tuple


class PR70ArrivalTests(unittest.TestCase):
    def test_only_missing_stage1_is_injected(self):
        current = Observation((Sample("J1::stage1"),))
        actual = Observation((
            Sample("J1::stage1"), Sample("J1::stage2"),
            Sample("J2::stage1"), Sample("J2::stage2"),
        ))
        updated, added = inject_public_stage1(current, actual)
        self.assertEqual(added, ["J2::stage1"])
        self.assertEqual([s.id for s in updated.samples],
                         ["J1::stage1", "J2::stage1"])

    def test_existing_stage1_not_duplicated(self):
        current = Observation((Sample("J1::stage1"),))
        actual = Observation((Sample("J1::stage1"),))
        updated, added = inject_public_stage1(current, actual)
        self.assertEqual(added, [])
        self.assertEqual(updated, current)


if __name__ == "__main__":
    unittest.main()
