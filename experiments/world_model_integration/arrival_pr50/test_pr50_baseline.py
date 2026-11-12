"""PR50 train-only Poisson and public prefix conditional arrival baseline contracts."""
import math
import unittest

from experiments.world_model_integration.arrival_pr50.pr50_baseline import (
    CANDIDATES,ANCHORS,HORIZONS,train_poisson,rate_at,
    poisson_count_nll,per_episode,metric
)


def episode(name,marks,ticks=75):
    counts=[0]*ticks
    events=[]
    for tick,mark in marks:
        counts[tick]+=1
        events.append({"tick":tick,"mark":mark})
    return {
        "id":name,"mechanism":"independent",
        "public_first_job_events":events,
        "public_count_by_tick":counts,
        "public_observed_ticks":ticks,
        "public_jobs":len(events)
    }


class PR50TrainTestContracts(unittest.TestCase):
    def test_poisson_global_uses_train_public_exposure_only(self):
        e1=episode("a",[(2,"w0"),(5,"w1")])
        e2=episode("b",[(1,"w0")],ticks=25)
        p=train_poisson([e1,e2])
        self.assertEqual(p["global_arrivals_per_tick"],3/100)
        self.assertAlmostEqual(sum(p["mark_probs"].values()),1)
        self.assertEqual(p["train_public_event_count"],3)

    def test_conditional_rate_never_reads_future(self):
        prior={"global_arrivals_per_tick":.2}
        prefix=[0,1,0,1,0,0,1,0,0,0]
        a=prefix+[0]*20
        b=prefix+[3]*20
        for name in CANDIDATES:
            self.assertEqual(rate_at(a,9,prior,name),rate_at(b,9,prior,name))
        self.assertEqual(rate_at(a,9,prior,"poisson_global"),.2)

    def test_never_scores_future_beyond_last_public_tick(self):
        e=episode("c",[(8,"w0"),(30,"w1")],ticks=44)
        p=train_poisson([e])
        self.assertEqual(len(per_episode(e,p,"poisson_global",20)),2)
        data=metric([e],p,"poisson_global",20)
        self.assertEqual(data["scenario_groups"],1)
        self.assertEqual(data["total_eligible_anchors"],2)

    def test_count_nll_is_proper_poisson_mass_score(self):
        self.assertAlmostEqual(poisson_count_nll(0,1.),1.)
        self.assertAlmostEqual(poisson_count_nll(2,2.),
                               2.-2.*math.log(2.)+math.log(2.))
        with self.assertRaises(ValueError):
            poisson_count_nll(2,0.)

    def test_horizons_and_selection_candidates_frozen(self):
        self.assertEqual(HORIZONS,(5,20))
        self.assertEqual(ANCHORS,(8,16,24,32,40))
        self.assertEqual(CANDIDATES,(
            "poisson_global","history_4","history_8","history_16"))


if __name__=="__main__":
    unittest.main()
