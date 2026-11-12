"""PR65 CPU unit tests: contract, causal input-only, conditional risk and output scope."""
import copy
import unittest

from experiments.world_model_integration.stochastic_pr65.pr65_public_device import (
    PublicStationHazardAdapter, visible_age_h4,
)


def dummy_input():
    def x(v):
        row = [0.] * 16
        row[6] = float(v)
        return row
    return {
        "public_tick": 15,
        "feature_last_observed_tick": 15,
        "graph": {"nodes": [
            {"kind": "station", "id": "B0", "device_available": True},
            {"kind": "station", "id": "B1", "device_available": False},
            {"kind": "task", "id": "t1"},
        ]},
        "history": [
            [x(1), x(1), x(1), x(1)],
            [x(1), x(1), x(0), x(0)],
            [x(0), x(0), x(0), x(0)],
        ],
        "history_visible_mask": [
            [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]
        ],
        "node_persistence": [[0.] * 6 for _ in range(3)],
        "input_source": "real_public",
        "public_map": {"static_map_digest": "test-public-map"},
    }


def fitted():
    return {
        "fit_only_on_train": True,
        "constant": {
            "up": {"n": 200, "events": 20, "hazard": 0.1},
            "down": {"n": 20, "events": 4, "hazard": 0.2},
        },
        "age_h4": {
            "up": {"4": {"n": 30, "events": 9, "hazard": 0.3}},
            "down": {"2": {"n": 8, "events": 3, "hazard": 0.4}},
        },
    }


class PublicStationHazardAdapterTests(unittest.TestCase):
    def test_public_age_matches_pr63_contract(self):
        inp = dummy_input()
        self.assertEqual(visible_age_h4(inp["history"][0],
                                        inp["history_visible_mask"][0], 1), 4)
        self.assertEqual(visible_age_h4(inp["history"][1],
                                        inp["history_visible_mask"][1], 0), 2)
        with self.assertRaises(ValueError):
            visible_age_h4(inp["history"][0], [0, 0, 0, 0], 1)

    def test_up_down_probability_and_unseen_age_fallback(self):
        inp = dummy_input()
        p = PublicStationHazardAdapter(fitted(), "constant").station_probabilities(inp)
        self.assertEqual([r.station for r in p], ["B0", "B1"])
        self.assertAlmostEqual(p[0].probability_next_available, 0.9)
        self.assertAlmostEqual(p[1].probability_next_available, 0.2)
        self.assertAlmostEqual(p[0].probability_transition, 0.1)
        self.assertAlmostEqual(p[1].probability_transition, 0.2)
        h = PublicStationHazardAdapter(fitted(), "age_h4").station_probabilities(inp)
        self.assertAlmostEqual(h[0].probability_next_available, 0.7)
        self.assertAlmostEqual(h[1].probability_next_available, 0.4)
        del inp["history_visible_mask"][1][0]
        with self.assertRaises(ValueError):
            PublicStationHazardAdapter(fitted(), "age_h4").station_probabilities(inp)

    def test_only_public_input_and_train_fitting_admitted(self):
        inp = dummy_input()
        for extra in ("target", "private_outage_start", "future_device_state"):
            bad = copy.deepcopy(inp)
            bad[extra] = [42]
            with self.assertRaises(ValueError):
                PublicStationHazardAdapter(fitted()).station_probabilities(bad)
        bad_params = fitted()
        bad_params["fit_only_on_train"] = False
        with self.assertRaises(ValueError):
            PublicStationHazardAdapter(bad_params)
        with self.assertRaises(ValueError):
            PublicStationHazardAdapter(fitted(), "clock4")
        before = PublicStationHazardAdapter(fitted()).station_probabilities(inp)
        other = copy.deepcopy(inp)
        other["public_tick"] = other["feature_last_observed_tick"] = 44
        self.assertEqual(before, PublicStationHazardAdapter(fitted()).station_probabilities(other))

    def test_only_station_device_channel_changes(self):
        import torch
        inp = dummy_input()
        nodes = torch.rand(3, 6, requires_grad=True)
        original = nodes.detach().clone()
        out = PublicStationHazardAdapter(fitted()).overlay_one_step(inp, nodes)
        self.assertAlmostEqual(float(out[0, 0]), 0.9, places=6)
        self.assertAlmostEqual(float(out[1, 0]), 0.2, places=6)
        unchanged = torch.ones(3, 6, dtype=torch.bool)
        unchanged[:2, 0] = False
        self.assertTrue(torch.equal(out[unchanged], original[unchanged]))
        out.sum().backward()
        self.assertIsNotNone(nodes.grad)
        self.assertEqual(float(nodes.grad[0, 0]), 0.0)
        self.assertEqual(float(nodes.grad[2, 0]), 1.0)

    def test_seeded_device_only_particle_proposals(self):
        a = PublicStationHazardAdapter(fitted(), "age_h4")
        inp = dummy_input()
        one = a.sample_device_event_proposals(inp, seed=112, particles=40)
        two = a.sample_device_event_proposals(inp, seed=112, particles=40)
        self.assertEqual(one, two)
        self.assertEqual(len(one), 40)
        self.assertTrue(all(set(row) == {"B0", "B1"} for row in one))
        self.assertTrue(all(set(row.values()) <= {0, 1} for row in one))
        with self.assertRaises(ValueError):
            a.sample_device_event_proposals(inp, seed=1, particles=0)


if __name__ == "__main__":
    unittest.main()
