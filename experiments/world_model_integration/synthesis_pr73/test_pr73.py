"""PR73 original deterministic-jitter mechanics and public-input bounds."""
import unittest

from experiments.business_constraints.tests.test_business import fixture
from business_transport.simulation import BusinessSimulator
from experiments.world_model_integration.synthesis_pr73.pr73_pilot import (
    alter_jitter, aggregate, PHASES,
)


class PR73OriginalMechanismTests(unittest.TestCase):
    def test_only_station_jitter_changes(self):
        base, cfg, world = fixture()
        altered = alter_jitter(world, 2)
        self.assertEqual(altered.rules, world.rules)
        self.assertEqual(altered.jobs, world.jobs)
        self.assertEqual(altered.scenario.robots, world.scenario.robots)
        self.assertEqual(altered.scenario.samples, world.scenario.samples)
        self.assertEqual(altered.scenario.grid, world.scenario.grid)
        self.assertEqual(altered.scenario.duration_jitter, world.scenario.duration_jitter)
        for sid, station in world.scenario.stations.items():
            new = altered.scenario.stations[sid]
            self.assertEqual(new.service_ticks, station.service_ticks)
            self.assertEqual(new.prior_ticks, station.prior_ticks)
            self.assertEqual(new.outages, station.outages)
            self.assertEqual(new.jitter_ticks, 2)
        self.assertEqual(alter_jitter(world, 0).scenario.stations["A0"].jitter_ticks, 0)

    def test_original_service_jitter_keyed_and_bounded(self):
        _, _, world = fixture()
        base = BusinessSimulator(alter_jitter(world, 0), "world", 731)
        variable = BusinessSimulator(alter_jitter(world, 2), "world", 731)
        variable_again = BusinessSimulator(alter_jitter(world, 2), "world", 731)
        differences = []
        for name in ("J1::stage1", "J2::stage1", "J9::stage1"):
            for phase in PHASES:
                x = base._service_duration(name, phase, "A0")
                y = variable._service_duration(name, phase, "A0")
                self.assertEqual(x, world.scenario.stations["A0"].service_ticks)
                self.assertIn(y-x, (0, 1, 2))
                self.assertEqual(
                    y, variable_again._service_duration(name, phase, "A0"))
                differences.append(y-x)
        self.assertGreater(len(set(differences)), 1)

    def test_model_no_leak_of_jitter_truth(self):
        a = [
            {"phase": "to_pickup", "station": "A0", "active_elapsed": 1,
             "event": 0, "rule_event": 0, "wall_elapsed": 1,
             "public_effective_mean_known": False, "query_width": 48},
            {"phase": "to_pickup", "station": "A0", "active_elapsed": 1,
             "event": 1, "rule_event": 0, "wall_elapsed": 1,
             "public_effective_mean_known": False, "query_width": 48},
        ]
        r = aggregate(a)
        self.assertEqual(r["coarse_strata_with_both_event_outcomes"], 1)
        self.assertEqual(r["per_phase"]["to_pickup"]["events"], 1)
        self.assertEqual(r["per_phase"]["to_pickup"]["rule_brier"], 0.5)

    def test_unsupported_mechanism_fails(self):
        _, _, world = fixture()
        with self.assertRaisesRegex(ValueError, "preregistered"):
            alter_jitter(world, 5)


if __name__ == "__main__":
    unittest.main()
