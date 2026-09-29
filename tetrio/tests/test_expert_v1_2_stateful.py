from __future__ import annotations

import unittest
import numpy as np
import torch

from tetrio.future.features import FEATURE_SIZE
from tetrio.network.model_v1_2 import StatefulResidualReranker
from tetrio.stateful.features import (
    STATEFUL_FEATURE_SIZE,
    encode_battle_state_row,
    normalize_stateful_features_numpy,
)


class ExpertV12StatefulTests(unittest.TestCase):
    def test_feature_encoding(self):
        x=encode_battle_state_row(
            raw_combo_before=2, raw_btb_before=3, previous_cleared=2,
            previous_t_spin="MINI", previous_attack=4, previous_garbage_cleared=1,
        )
        self.assertEqual(x.shape,(STATEFUL_FEATURE_SIZE,))
        self.assertEqual(float(x[3]),1.0); self.assertEqual(float(x[4]),1.0)
        y=normalize_stateful_features_numpy(x)
        self.assertTrue(np.isfinite(y).all())

    def test_e00_zero_init_is_exact_zero(self):
        m=StatefulResidualReranker(hidden_size=16,max_adjustment=1.0)
        b,k=3,5
        out=m(
            v11_scores=torch.randn(b,k),
            raw_future_features=torch.randn(b,k,FEATURE_SIZE),
            candidate_use_hold=torch.zeros(b,k,dtype=torch.bool),
            battle_state=torch.randn(b,STATEFUL_FEATURE_SIZE),
            mask=torch.ones(b,k,dtype=torch.bool),
        )
        self.assertTrue(torch.equal(out,torch.zeros_like(out)))

    def test_zero_state_stays_zero_after_training_weights_change(self):
        m=StatefulResidualReranker(hidden_size=16,max_adjustment=1.0)
        with torch.no_grad():
            for p in m.parameters():
                p.uniform_(-0.5,0.5)
        b,k=2,4
        out=m(
            v11_scores=torch.randn(b,k),
            raw_future_features=torch.randn(b,k,FEATURE_SIZE),
            candidate_use_hold=torch.randint(0,2,(b,k),dtype=torch.bool),
            battle_state=torch.zeros(b,STATEFUL_FEATURE_SIZE),
            mask=torch.ones(b,k,dtype=torch.bool),
        )
        self.assertTrue(torch.equal(out,torch.zeros_like(out)))

if __name__=="__main__": unittest.main()
