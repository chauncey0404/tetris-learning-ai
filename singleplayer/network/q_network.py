from __future__ import annotations

import torch.nn as nn

from tetris_ai.networks import CandidateQNetwork

STATE_SIZE = 243
CANDIDATE_SIZE = 215


class ObservableSafeQNetwork(CandidateQNetwork):
    """V8 observable-safe adapter over the shared candidate-Q architecture."""

    def __init__(self):
        super().__init__(
            state_size=STATE_SIZE,
            candidate_size=CANDIDATE_SIZE,
            teacher_score_scale=1000.0,
            teacher_rank_scale=4.0,
        )


def copy_matching_state_encoder_weights(model, old_state_dict):
    new_state = model.state_dict()
    copied = []
    for key, value in old_state_dict.items():
        if not key.startswith("state_encoder.") or key not in new_state:
            continue
        if tuple(new_state[key].shape) != tuple(value.shape):
            continue
        new_state[key] = value.detach().cpu().clone()
        copied.append(key)
    model.load_state_dict(new_state)
    nn.init.zeros_(model.q_head.weight)
    nn.init.zeros_(model.q_head.bias)
    return copied


ObservableQNetwork = ObservableSafeQNetwork
