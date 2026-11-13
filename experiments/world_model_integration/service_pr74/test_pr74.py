"""PR74: strict reuse of existing original Deep Sets 48d model and source gates."""
import unittest

import torch

from lab_transport.business_session_learning import BusinessSessionEventModel
from experiments.world_model_integration.service_pr74.pr74_cpu_training import (
    EPOCHS, SEED, BATCH_SIZE, LR, JITTER, scene_macro,
)


class PR74FrozenArchitectureTests(unittest.TestCase):
    def test_reuse_original_model_tensor_contract(self):
        net=BusinessSessionEventModel(query_width=48).cpu()
        x=torch.zeros((3,5,40))
        m=torch.ones((3,5))
        q=torch.zeros((3,48))
        with torch.no_grad():
            y=net(x,m,q)
        self.assertEqual(tuple(y.shape),(3,))
        self.assertTrue(torch.isfinite(y).all())
        with self.assertRaisesRegex(ValueError, "invalid per-session query"):
            net(x,m,torch.zeros(3,49))

    def test_single_seed_cpu_study_limits(self):
        self.assertEqual((EPOCHS,SEED,BATCH_SIZE,LR,JITTER),
                         (20,1810,64,0.001,2))

    def test_scene_aggregation_uses_scenes_not_steps(self):
        dev=[{"label":0},{"label":1},{"label":1}]
        meta=[
            {"scene":"A","event":0,"rule_event":0},
            {"scene":"A","event":1,"rule_event":0},
            {"scene":"B","event":1,"rule_event":1},
        ]
        data=scene_macro(dev,[0.5,0.5,0.8],meta,None)
        self.assertEqual(data["n_scenes"],2)
        self.assertAlmostEqual(data["scene_macro_model_brier"],(0.25+0.04)/2)
        self.assertAlmostEqual(data["scene_macro_rule_brier"],(0.5+0.0)/2)


if __name__=="__main__":
    unittest.main()
