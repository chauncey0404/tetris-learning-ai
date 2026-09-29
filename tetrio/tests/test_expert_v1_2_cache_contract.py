from __future__ import annotations
import unittest
import numpy as np
from tetrio.network.stateful_cache import validate_stateful_pair
from tetrio.stateful.features import STATEFUL_FEATURE_SIZE

class ExpertV12CacheContractTests(unittest.TestCase):
    def test_identity_match(self):
        future={"expert_local":np.zeros(2,dtype=np.int16),"game_id":np.array([1,2]),"subframe":np.array([10,20])}
        state={"state_features":np.zeros((2,STATEFUL_FEATURE_SIZE),dtype=np.float32),"game_id":np.array([1,2]),"subframe":np.array([10,20])}
        validate_stateful_pair(future,state)
    def test_identity_mismatch_fails(self):
        future={"expert_local":np.zeros(1,dtype=np.int16),"game_id":np.array([1]),"subframe":np.array([10])}
        state={"state_features":np.zeros((1,STATEFUL_FEATURE_SIZE),dtype=np.float32),"game_id":np.array([1]),"subframe":np.array([11])}
        with self.assertRaises(RuntimeError): validate_stateful_pair(future,state)
if __name__=="__main__": unittest.main()
