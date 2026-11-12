"""PR50 strict public arrival labels, EasyTPP adapter and frozen independent splits."""
import json
import unittest
from unittest.mock import patch

from experiments.world_model_integration.arrival_pr50.pr50_scenario import (
    PROTOCOL,validate_protocol,_private_arrivals,MECHANISMS,MARKS,true_business,
)
from experiments.world_model_integration.arrival_pr50.pr50_public_dataset import (
    PublicArrivals,to_easytpp,one,
)


class PR50ArrivalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=json.loads(PROTOCOL.read_text())

    def test_independent_cohorts_and_latent_generators(self):
        self.assertTrue(validate_protocol(self.p))
        self.assertEqual(len(self.p["cohort"]),144)
        self.assertEqual(len({x["seed"] for x in self.p["cohort"]}),144)
        for m in MECHANISMS:
            selected=[x for x in self.p["cohort"] if x["mechanism"]==m]
            self.assertEqual(len(selected),48)
            self.assertEqual(sum(x["split"]=="train" for x in selected),24)
            self.assertEqual(sum(x["split"]=="test" for x in selected),8)

    def test_public_adapter_is_bare_event_sequence_no_private_mechanism(self):
        d={"public_first_job_events":[{"tick":0,"mark":"w0"},
                                      {"tick":0,"mark":"w1"},
                                      {"tick":5,"mark":"w0"}],
           "seed":123,"split":"test","mechanism":"hidden_bursts",
           "private_schedule_count_for_integrity_only":20}
        item=to_easytpp(d)
        self.assertEqual(set(item),{"time_since_start","time_since_last_event",
                                    "type_event","dim_process"})
        self.assertEqual(item["time_since_start"],[0.0,0.0,5.0])
        self.assertEqual(item["time_since_last_event"],[0.0,0.0,5.0])
        self.assertEqual(item["type_event"],[0,1,0])
        self.assertEqual(item["dim_process"],2)

    def test_generator_future_schedule_stays_private_and_marks_valid(self):
        for m in MECHANISMS:
            spec=next(x for x in self.p["cohort"] if x["mechanism"]==m)
            jobs=_private_arrivals(spec,self.p)
            self.assertTrue(jobs)
            self.assertTrue(all(j["release_tick"]<75 and j["workflow"] in MARKS for j in jobs))
            self.assertEqual(len({j["id"] for j in jobs}),len(jobs))

    def test_true_simulator_public_event_smoke_has_no_hidden_future(self):
        spec=next(s for s in self.p["cohort"] if s["mechanism"]=="periodic"
                  and s["split"]=="train")
        row=one(spec,self.p)
        self.assertGreater(row["public_jobs"],0)
        self.assertEqual(sum(row["public_count_by_tick"]),row["public_jobs"])
        self.assertTrue(all(e["tick"]<row["public_observed_ticks"]
                            and e["mark"] in MARKS
                            for e in row["public_first_job_events"]))
        self.assertTrue(row["public_jobs"]<=row["private_schedule_count_for_integrity_only"])
        for forbidden in ("future_job_release_schedule","private_next_release",
                          "hidden_latent_state"):
            self.assertNotIn(forbidden,row)
        self.assertEqual(len(to_easytpp(row)["type_event"]),row["public_jobs"])

    def test_shift_test_never_used_for_training(self):
        self.assertEqual({r["split"] for r in self.p["cohort"]},
                         {"train","validation","test","shift_test"})
        for p in self.p["cohort"]:
            self.assertEqual(p["shift"],p["split"]=="shift_test")


if __name__=="__main__":
    unittest.main()
