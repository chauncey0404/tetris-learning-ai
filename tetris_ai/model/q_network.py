import torch
import torch.nn as nn


STATE_SIZE = 243
CANDIDATE_SIZE = 215


class ObservableSafeQNetwork(nn.Module):
    """
    Q(s, a) network for V8.4.

    IMPORTANT:
    - `state` is the currently observable 243-d state.
    - `candidates` is a 215-d legal action-afterstate representation that
      contains ONLY the deterministic board result + action metadata.
    - The full preview successor state is deliberately NOT an input.
    """

    def __init__(self):
        super().__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_SIZE, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )

        self.candidate_encoder = nn.Sequential(
            nn.Linear(CANDIDATE_SIZE, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )

        # state latent 32 + candidate latent 32 + reward/score/rank 3 = 67
        self.joint = nn.Sequential(
            nn.Linear(67, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )

        self.q_head = nn.Linear(32, 1)

        # Teacher remains the default at initialization because all Q values tie.
        nn.init.zeros_(self.q_head.weight)
        nn.init.zeros_(self.q_head.bias)

    def forward(
        self,
        *,
        state,
        candidates,
        rewards,
        teacher_scores,
        teacher_ranks,
    ):
        if state.ndim != 2 or state.shape[-1] != STATE_SIZE:
            raise ValueError(
                f"state must be (B,{STATE_SIZE}); got {tuple(state.shape)}"
            )
        if candidates.ndim != 3 or candidates.shape[-1] != CANDIDATE_SIZE:
            raise ValueError(
                "candidates must be "
                f"(B,K,{CANDIDATE_SIZE}); got {tuple(candidates.shape)}"
            )

        batch_size, candidate_count, _ = candidates.shape

        for name, value in (
            ("rewards", rewards),
            ("teacher_scores", teacher_scores),
            ("teacher_ranks", teacher_ranks),
        ):
            if value.shape != (batch_size, candidate_count):
                raise ValueError(
                    f"{name} must be {(batch_size, candidate_count)}; "
                    f"got {tuple(value.shape)}"
                )

        state_latent = self.state_encoder(state)
        state_latent = state_latent.unsqueeze(1).expand(
            -1, candidate_count, -1
        )

        flat_candidates = candidates.reshape(
            batch_size * candidate_count,
            CANDIDATE_SIZE,
        )
        candidate_latent = self.candidate_encoder(flat_candidates)
        candidate_latent = candidate_latent.reshape(
            batch_size,
            candidate_count,
            -1,
        )

        # Keep approximately the same scalar scaling used by V8.x.
        reward_scalar = rewards.unsqueeze(-1)
        score_scalar = (teacher_scores / 1000.0).unsqueeze(-1)
        rank_scalar = (teacher_ranks / 4.0).unsqueeze(-1)

        joint_input = torch.cat(
            (
                state_latent,
                candidate_latent,
                reward_scalar,
                score_scalar,
                rank_scalar,
            ),
            dim=-1,
        )

        latent = self.joint(joint_input)
        return self.q_head(latent).squeeze(-1)


def copy_matching_state_encoder_weights(model, old_state_dict):
    """
    Optional migration helper.

    Only matching `state_encoder.*` tensors are copied. Candidate encoder,
    joint layers and Q head are never copied from a pre-V8.4 model.
    This function is OFF by default in the trainer.
    """
    new_state = model.state_dict()
    copied = []

    for key, value in old_state_dict.items():
        if not key.startswith("state_encoder."):
            continue
        if key not in new_state:
            continue
        if tuple(new_state[key].shape) != tuple(value.shape):
            continue
        new_state[key] = value.detach().cpu().clone()
        copied.append(key)

    model.load_state_dict(new_state)

    # Always reset Q head so the migrated model starts Teacher-safe.
    nn.init.zeros_(model.q_head.weight)
    nn.init.zeros_(model.q_head.bias)

    return copied


# Stable, version-free public name.
ObservableQNetwork = ObservableSafeQNetwork
