from dataclasses import dataclass

from singleplayer.game.placement import enumerate_state_placements, piece_for_placement
from singleplayer.heuristic.features import BoardFeaturesV2
from singleplayer.heuristic.i_dependency import IDependencyAnalysis
from singleplayer.heuristic.scoring import score_candidate_v2

@dataclass(frozen=True)
class TeacherDecisionV2:
    candidate: object

    played_piece: str

    score: float

    features: BoardFeaturesV2

    i_dependency: IDependencyAnalysis


class HeuristicTeacherV2:

    name = (
        "HEURISTIC_TEACHER_V2_1_"
        "TETRIS_POTENTIAL_I_DEPENDENCY"
    )

    def rank(
        self,
        state,
    ):
        """
        回傳所有 candidate，
        依 Teacher 分數由高到低排序。

        choose() 仍然只是拿第一名。

        Gym / 真正遊戲 Adapter 可以從 rank()
        中挑出「實際可到達」的最高分 placement。
        """

        candidates = (
            enumerate_state_placements(
                state
            )
        )

        if not candidates:
            return []

        scored = []

        for candidate in candidates:

            if candidate.top_out:
                continue

            (
                score,
                features,
                i_info,
            ) = score_candidate_v2(
                state,
                candidate,
            )

            played_piece = (
                piece_for_placement(
                    state,
                    candidate.action,
                )
            )

            scored.append(
                TeacherDecisionV2(
                    candidate=candidate,
                    played_piece=played_piece,
                    score=score,
                    features=features,
                    i_dependency=i_info,
                )
            )

        scored.sort(
            key=lambda decision: (
                decision.score,
                decision.candidate.lines_cleared,
                -int(
                    decision
                    .candidate
                    .action
                    .use_hold
                ),
                -decision
                .candidate
                .action
                .rotation,
                -decision
                .candidate
                .action
                .x,
            ),
            reverse=True,
        )

        return scored

    def choose(
        self,
        state,
    ):

        ranked = self.rank(
            state
        )

        if not ranked:
            return None

        return ranked[0]
