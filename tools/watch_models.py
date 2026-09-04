from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Optional

import numpy as np
import torch


# Allow both:
#   python -m tools.watch_models ...
# and:
#   python tools\watch_models.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from tetris_ai.game.core import GymTetrisAdapter
from tetris_ai.game.executor import execute_placement
from tetris_ai.game.placement import (
    get_rotations,
    piece_for_placement,
    simulate_placement,
)
from tetris_ai.heuristic.teacher import HeuristicTeacherV2
from tetris_ai.model.candidates import compact_candidate_arrays
from tetris_ai.model.q_network import ObservableSafeQNetwork
from tetris_ai.model.state_encoder import encode_state
from tetris_ai.policy.confidence import normalized_margin_choice
from tetris_ai.policy.legacy_raw_margin import conservative_choice
from tetris_ai.policy.successor import preview_top_k_successors


TOP_K = 4

LINE_VALUE = {
    0: 0,
    1: 900,
    2: 2000,
    3: 3300,
    4: 6000,
}

PROTECTED_FINAL_SEEDS = range(6, 21)

DEFAULT_NORMALIZED_GATE = 0.600
DEFAULT_RAW_GATE = 0.060

SPEED_PRESETS = {
    1: 0.5,
    2: 1.0,
    3: 2.0,
    4: 4.0,
    5: 6.0,
    6: 10.0,
    7: 15.0,
    8: 25.0,
    9: 60.0,
}

# Model-panel accent colors. These are NOT tetromino colors.
PALETTE = [
    (58, 194, 255),
    (255, 184, 76),
    (129, 219, 129),
    (218, 126, 255),
    (255, 115, 115),
    (89, 224, 199),
    (255, 225, 91),
    (148, 166, 255),
]

# Tetris Guideline tetromino color convention:
# I = cyan, J = blue, L = orange, O = yellow,
# S = green, T = purple, Z = red.
TETROMINO_COLORS = {
    "I": (0, 240, 240),
    "J": (0, 80, 240),
    "L": (240, 160, 0),
    "O": (240, 240, 0),
    "S": (0, 220, 0),
    "T": (160, 0, 240),
    "Z": (240, 0, 0),
}

PIECE_ID_TO_NAME = {
    2: "I",
    3: "O",
    4: "T",
    5: "S",
    6: "Z",
    7: "J",
    8: "L",
}

BG = (18, 20, 24)
PANEL_BG = (27, 30, 36)
PANEL_BORDER = (64, 69, 80)
GRID = (48, 53, 62)
EMPTY = (21, 24, 29)
TEXT = (232, 235, 240)
MUTED = (160, 166, 178)
GOOD = (105, 222, 143)
WARN = (255, 194, 88)
BAD = (255, 112, 112)
TEACHER_COLOR = (173, 181, 255)


def short_steps(value: int) -> str:
    if value < 0:
        return "?"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(value)


def safe_stem(path: Path, max_len: int = 38) -> str:
    stem = path.stem
    if len(stem) <= max_len:
        return stem
    return stem[: max_len - 1] + "…"


def choose_device(name: str) -> torch.device:
    name = str(name).lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available.")
    return torch.device(name)


def board_metrics(board: np.ndarray) -> tuple[int, int]:
    board = np.asarray(board, dtype=np.uint8).reshape(20, 10)

    occupied_rows = np.where(np.any(board != 0, axis=1))[0]
    height = 0 if occupied_rows.size == 0 else int(20 - occupied_rows[0])

    holes = 0
    for col in range(10):
        column = board[:, col]
        filled = np.where(column != 0)[0]
        if filled.size == 0:
            continue
        top = int(filled[0])
        holes += int(np.count_nonzero(column[top:] == 0))

    return height, holes


def piece_color(piece: Optional[str]) -> tuple[int, int, int]:
    if piece is None:
        return MUTED
    return TETROMINO_COLORS.get(str(piece), (185, 190, 200))


def brighten(color: tuple[int, int, int], amount: int = 42) -> tuple[int, int, int]:
    return tuple(min(255, int(c) + amount) for c in color)


def darken(color: tuple[int, int, int], factor: float = 0.58) -> tuple[int, int, int]:
    return tuple(max(0, int(c * factor)) for c in color)


def visual_hold_and_queue(state, action) -> tuple[Optional[str], tuple[str, ...]]:
    """
    Return the Hold and Next state that should be shown while the selected
    placement is visually falling.

    The actual game is still executed by the validated hard-drop executor.
    This function only keeps the viewer visually consistent with Hold use.
    """
    hold = state.hold_piece
    queue = tuple(state.next_pieces)

    if not bool(action.use_hold):
        return hold, queue

    # Current piece enters Hold.
    new_hold = state.current_piece

    # If Hold already contained a piece, the queue does not advance.
    if state.hold_piece is not None:
        return new_hold, queue

    # Empty Hold: next_pieces[0] becomes the active falling tetromino.
    if queue:
        return new_hold, queue[1:]

    return new_hold, queue


@dataclass
class LoadedPolicy:
    spec: str
    label: str
    is_teacher: bool
    model: Optional[ObservableSafeQNetwork]
    gate: float
    gate_semantics: str
    env_steps: int = -1
    gradient_steps: int = -1
    path: Optional[Path] = None

    @property
    def gate_short(self) -> str:
        if self.is_teacher:
            return "Teacher only"
        kind = "norm" if self.gate_semantics == "normalized_q_margin" else "raw"
        return f"{kind} gate={self.gate:.3f}"


def infer_gate_semantics(checkpoint: dict) -> str:
    explicit = checkpoint.get("gate_semantics")
    if explicit is not None:
        explicit = str(explicit)
        if explicit in {"normalized_q_margin", "raw_q_gap"}:
            return explicit
        raise RuntimeError(f"Unsupported gate_semantics in checkpoint: {explicit}")

    # V8.7+ normalized checkpoints explicitly recorded gate_semantics.
    # A checkpoint without that metadata is treated as the older raw-Q-gap
    # policy unless it has an explicit normalized_gate field.
    if "normalized_gate" in checkpoint:
        return "normalized_q_margin"

    return "raw_q_gap"


def infer_gate(checkpoint: dict, semantics: str) -> float:
    if semantics == "normalized_q_margin":
        return float(
            checkpoint.get(
                "normalized_gate",
                checkpoint.get("target_gate", DEFAULT_NORMALIZED_GATE),
            )
        )

    return float(
        checkpoint.get(
            "q_gate",
            checkpoint.get("target_gate", DEFAULT_RAW_GATE),
        )
    )


def load_policy(
    spec: str,
    *,
    label: Optional[str],
    device: torch.device,
    gate_override: Optional[float],
    semantics_override: str,
) -> LoadedPolicy:
    if spec.lower() in {"teacher", "@teacher"}:
        return LoadedPolicy(
            spec=spec,
            label=label or "Teacher",
            is_teacher=True,
            model=None,
            gate=0.0,
            gate_semantics="teacher",
        )

    path = Path(spec)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError(
            f"Checkpoint has no model_state_dict: {path}\n"
            "watch_models currently supports the ObservableSafeQNetwork "
            "checkpoint family (V8.4+)."
        )

    model = ObservableSafeQNetwork()
    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(
            f"Checkpoint architecture is not compatible with "
            f"ObservableSafeQNetwork: {path}\n{exc}"
        ) from exc

    model.to(device)
    model.eval()

    if semantics_override == "auto":
        semantics = infer_gate_semantics(checkpoint)
    else:
        semantics = semantics_override

    gate = infer_gate(checkpoint, semantics)
    if gate_override is not None:
        gate = float(gate_override)

    env_steps = int(checkpoint.get("env_steps", -1))
    gradient_steps = int(checkpoint.get("gradient_steps", -1))

    auto_label = safe_stem(path)
    if env_steps >= 0:
        auto_label = f"{auto_label} [{short_steps(env_steps)}]"

    return LoadedPolicy(
        spec=spec,
        label=label or auto_label,
        is_teacher=False,
        model=model,
        gate=float(gate),
        gate_semantics=semantics,
        env_steps=env_steps,
        gradient_steps=gradient_steps,
        path=path,
    )


@dataclass
class DecisionInfo:
    piece: str = "-"
    action_text: str = "-"
    source: str = "-"
    chosen_index: int = 0
    confidence: float = 0.0
    q_values: list[float] = field(default_factory=list)
    teacher_score: float = 0.0
    lines_cleared: int = 0


@dataclass
class VisualDrop:
    """
    Purely visual "human-like" movement animation.

    The AI still decides only the final canonical placement, and the validated
    executor still performs the real rotate -> horizontal -> hard-drop action.
    This viewer reconstructs a readable animation:

      spawn centered at rotation 0
          -> rotate / move horizontally near the top
          -> fall vertically into the exact chosen landing cell

    Therefore the animation never changes model behavior or Race fairness.
    """
    board_ids: np.ndarray
    piece: str
    target_rotation: int
    target_x: int
    landing_y: int
    hold_piece: Optional[str]
    next_pieces: tuple[str, ...]
    started_at: float = field(default_factory=time.perf_counter)
    duration: float = 0.50

    def _rotations(self) -> dict[int, np.ndarray]:
        return {
            int(k): np.asarray(v, dtype=np.uint8)
            for k, v in dict(get_rotations(self.piece)).items()
        }

    @property
    def target_shape(self) -> np.ndarray:
        rotations = self._rotations()
        return rotations[self.target_rotation]

    @property
    def spawn_shape(self) -> np.ndarray:
        rotations = self._rotations()
        return rotations.get(0, rotations[min(rotations)])

    @property
    def spawn_x(self) -> int:
        # Guideline-like visual spawn: horizontally centered.
        shape = self.spawn_shape
        return max(0, (10 - int(shape.shape[1])) // 2)

    @property
    def spawn_y(self) -> int:
        # Begin partly above the visible playfield, like a normal Tetris spawn.
        return -max(1, int(self.spawn_shape.shape[0]) - 1)

    def progress(self, now: Optional[float] = None) -> float:
        if now is None:
            now = time.perf_counter()
        if self.duration <= 0:
            return 1.0
        return max(0.0, min(1.0, (now - self.started_at) / self.duration))

    def _rotation_for_adjust_progress(self, p: float) -> int:
        rotations = self._rotations()
        if self.target_rotation in rotations and self.target_rotation == 0:
            return 0

        # The canonical rotation contract uses repeated clockwise quarter-turns.
        # For pieces with only 2 unique rotations, target_rotation is normally
        # 0 or 1. For T/J/L it can be 0..3.
        target = int(self.target_rotation)
        if target <= 0:
            return 0

        steps_done = min(target, int(p * (target + 1)))
        candidate = steps_done

        if candidate in rotations:
            return candidate

        # Defensive fallback for a piece exposing a sparse rotation-key set.
        keys = sorted(rotations)
        usable = [k for k in keys if k <= candidate]
        return usable[-1] if usable else keys[0]

    def pose(
        self,
        now: Optional[float] = None,
    ) -> tuple[np.ndarray, int, int, float]:
        """
        Return (shape, x, y, progress) for the visible active tetromino.

        During the first ~48% of the animation, the piece descends only a few
        rows while visibly rotating and moving from center toward its chosen x.
        It then falls vertically in its final orientation.
        """
        p = self.progress(now)
        rotations = self._rotations()

        adjust_end = 0.48
        if p < adjust_end:
            a = p / adjust_end

            rot = self._rotation_for_adjust_progress(a)
            shape = rotations[rot]

            spawn_x = self.spawn_x
            x_float = spawn_x + (self.target_x - spawn_x) * a
            x = int(round(x_float))
            x = max(0, min(10 - int(shape.shape[1]), x))

            # Keep adjustment high enough that the animation reads as legal
            # spawn movement even when the stack is tall.
            safe_adjust_y = max(
                self.spawn_y,
                min(3, int(self.landing_y) - 2),
            )
            y_float = self.spawn_y + (safe_adjust_y - self.spawn_y) * a
            y = int(round(y_float))
            return shape, x, y, p

        # Final orientation / x is locked; now perform the visible gravity drop.
        shape = rotations[self.target_rotation]
        x = int(self.target_x)

        safe_adjust_y = max(
            self.spawn_y,
            min(3, int(self.landing_y) - 2),
        )
        fall_p = (p - adjust_end) / (1.0 - adjust_end)
        # Accelerating gravity-like motion.
        eased = fall_p * fall_p
        y_float = safe_adjust_y + (self.landing_y - safe_adjust_y) * eased
        y = int(round(y_float))
        return shape, x, y, p


@dataclass
class SessionStats:
    pieces: int = 0
    line_counts: dict[int, int] = field(
        default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0}
    )
    interventions: int = 0
    current_height: int = 0
    max_height: int = 0
    holes: int = 0
    max_holes: int = 0
    height_sum: float = 0.0

    @property
    def lines(self) -> int:
        return sum(k * v for k, v in self.line_counts.items())

    @property
    def tetrises(self) -> int:
        return self.line_counts[4]

    @property
    def value(self) -> int:
        return sum(LINE_VALUE[k] * v for k, v in self.line_counts.items())

    @property
    def switch_rate(self) -> float:
        return 0.0 if self.pieces == 0 else self.interventions / self.pieces

    @property
    def avg_height(self) -> float:
        return 0.0 if self.pieces == 0 else self.height_sum / self.pieces


class ModelSession:
    def __init__(
        self,
        policy: LoadedPolicy,
        *,
        seed: int,
        max_pieces: int,
        top_k: int,
        device: torch.device,
        teacher: HeuristicTeacherV2,
    ):
        self.policy = policy
        self.seed = int(seed)
        # 0 means unlimited: play until the environment reports game over.
        self.max_pieces = int(max_pieces)
        self.top_k = int(top_k)
        self.device = device
        self.teacher = teacher

        self.adapter = GymTetrisAdapter()
        self.state = None
        self.state_features = None

        self.stats = SessionStats()
        self.last = DecisionInfo()
        self.done = False
        self.game_over = False
        self.done_reason = ""
        self.last_exception = ""
        self.visual_drop: Optional[VisualDrop] = None

        self.reset(self.seed)

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self.state = self.adapter.reset(seed=self.seed)
        self.adapter.raw.gravity_enabled = False
        self.state_features = encode_state(self.state).astype(
            np.float32,
            copy=True,
        )

        h, holes = board_metrics(self.state.board)
        self.stats = SessionStats(
            current_height=h,
            max_height=h,
            holes=holes,
            max_holes=holes,
        )
        self.last = DecisionInfo()
        self.done = False
        self.game_over = False
        self.done_reason = ""
        self.last_exception = ""
        self.visual_drop = None

    def locked_board_ids(self) -> np.ndarray:
        """Return the 20x10 locked Gym board with original tetromino IDs."""
        raw_board = np.asarray(self.adapter.raw.board)
        padding = int(self.adapter.raw.padding)
        height = int(self.adapter.raw.height)
        width = int(self.adapter.raw.width)

        playfield = raw_board[
            0:height,
            padding:padding + width,
        ]
        return np.asarray(playfield, dtype=np.int16).copy()

    @torch.inference_mode()
    def _q_values(self, successors) -> np.ndarray:
        assert self.policy.model is not None

        candidates, rewards, scores, ranks = compact_candidate_arrays(successors)

        state_tensor = torch.from_numpy(
            np.asarray(self.state_features, dtype=np.float32)
        ).to(self.device).unsqueeze(0)

        candidate_tensor = torch.from_numpy(candidates).to(self.device).unsqueeze(0)
        reward_tensor = torch.from_numpy(rewards).to(self.device).unsqueeze(0)
        score_tensor = torch.from_numpy(scores).to(self.device).unsqueeze(0)
        rank_tensor = torch.from_numpy(ranks).to(self.device).unsqueeze(0)

        q = self.policy.model(
            state=state_tensor,
            candidates=candidate_tensor,
            rewards=reward_tensor,
            teacher_scores=score_tensor,
            teacher_ranks=rank_tensor,
        )[0]

        return q.detach().float().cpu().numpy()

    def _choose_index(self, q_values: np.ndarray) -> tuple[int, float]:
        if self.policy.gate_semantics == "normalized_q_margin":
            return normalized_margin_choice(
                q_values,
                self.policy.gate,
            )

        if self.policy.gate_semantics == "raw_q_gap":
            return conservative_choice(
                q_values,
                self.policy.gate,
            )

        raise RuntimeError(
            f"Unknown gate semantics: {self.policy.gate_semantics}"
        )

    def step(self) -> None:
        if self.done:
            return

        if self.max_pieces > 0 and self.stats.pieces >= self.max_pieces:
            self.done = True
            self.done_reason = "LIMIT"
            return

        try:
            piece_before = str(self.state.current_piece)
            pre_state = self.state
            pre_board_ids = self.locked_board_ids()

            successors = preview_top_k_successors(
                adapter=self.adapter,
                teacher=self.teacher,
                state=self.state,
                top_k=self.top_k,
            )

            if not successors:
                self.done = True
                self.game_over = True
                self.done_reason = "NO SUCCESSOR"
                return

            if self.policy.is_teacher:
                chosen_index = 0
                confidence = 0.0
                q_values = np.zeros(len(successors), dtype=np.float32)
            else:
                q_values = self._q_values(successors)
                chosen_index, confidence = self._choose_index(q_values)

            if not 0 <= chosen_index < len(successors):
                raise RuntimeError(
                    f"Chosen candidate {chosen_index} out of range "
                    f"for {len(successors)} successors."
                )

            chosen = successors[chosen_index]
            action = chosen.action

            if chosen_index != 0:
                self.stats.interventions += 1

            # Build an exact canonical visual drop before mutating the Gym env.
            # This does not affect policy/gameplay behavior.
            visual_drop = None
            try:
                falling_piece = str(piece_for_placement(pre_state, action))
                preview_result = simulate_placement(
                    board=pre_state.board,
                    piece=falling_piece,
                    rotation=int(action.rotation),
                    x=int(action.x),
                    use_hold=bool(action.use_hold),
                )
                rotations = dict(get_rotations(falling_piece))
                shape = rotations.get(int(action.rotation))
                if preview_result is not None and shape is not None:
                    visual_hold, visual_queue = visual_hold_and_queue(
                        pre_state,
                        action,
                    )
                    visual_drop = VisualDrop(
                        board_ids=pre_board_ids,
                        piece=falling_piece,
                        target_rotation=int(action.rotation),
                        target_x=int(action.x),
                        landing_y=int(preview_result.landing_y),
                        hold_piece=visual_hold,
                        next_pieces=tuple(visual_queue),
                    )
            except Exception:
                # Viewer animation is optional. Never let it alter evaluation.
                visual_drop = None

            result = execute_placement(
                self.adapter,
                action,
            )

            self.state = result["state"]
            self.state_features = encode_state(self.state).astype(
                np.float32,
                copy=True,
            )

            lines = int(result["info"].get("lines_cleared", 0))
            if lines in self.stats.line_counts:
                self.stats.line_counts[lines] += 1

            self.stats.pieces += 1
            h, holes = board_metrics(self.state.board)
            self.stats.current_height = h
            self.stats.max_height = max(self.stats.max_height, h)
            self.stats.holes = holes
            self.stats.max_holes = max(self.stats.max_holes, holes)
            self.stats.height_sum += h

            action_text = (
                f"{'H ' if bool(action.use_hold) else ''}"
                f"R{int(action.rotation)} X{int(action.x)}"
            )
            source = (
                "Teacher"
                if chosen_index == 0
                else f"Q -> #{chosen_index + 1}"
            )

            self.last = DecisionInfo(
                piece=piece_before,
                action_text=action_text,
                source=source,
                chosen_index=int(chosen_index),
                confidence=float(confidence),
                q_values=[float(x) for x in q_values],
                teacher_score=float(chosen.teacher_score),
                lines_cleared=lines,
            )
            self.visual_drop = visual_drop

            if bool(result["terminated"]):
                self.done = True
                self.game_over = True
                self.done_reason = "GAME OVER"

            elif bool(result["truncated"]):
                self.done = True
                self.game_over = False
                self.done_reason = "TRUNCATED"

            elif self.max_pieces > 0 and self.stats.pieces >= self.max_pieces:
                self.done = True
                self.done_reason = "LIMIT"

        except Exception as exc:
            self.done = True
            self.game_over = True
            self.done_reason = "ERROR"
            self.last_exception = f"{type(exc).__name__}: {exc}"

    def result_dict(self) -> dict:
        return {
            "label": self.policy.label,
            "checkpoint": (
                None if self.policy.path is None else str(self.policy.path)
            ),
            "seed": self.seed,
            "gate": self.policy.gate,
            "gate_semantics": self.policy.gate_semantics,
            "env_steps": self.policy.env_steps,
            "pieces": self.stats.pieces,
            "lines": self.stats.lines,
            "tetrises": self.stats.tetrises,
            "value": self.stats.value,
            "interventions": self.stats.interventions,
            "switch_rate": self.stats.switch_rate,
            "height": self.stats.current_height,
            "avg_height": self.stats.avg_height,
            "max_height": self.stats.max_height,
            "holes": self.stats.holes,
            "max_holes": self.stats.max_holes,
            "game_over": self.game_over,
            "done_reason": self.done_reason,
            "error": self.last_exception,
        }

    def close(self) -> None:
        self.adapter.close()


def race_rank(session: ModelSession) -> tuple:
    # Survival first, then official-like productivity/value, then safer stack.
    return (
        session.stats.pieces,
        session.stats.value,
        session.stats.lines,
        session.stats.tetrises,
        -session.stats.max_height,
        -session.stats.max_holes,
    )


def parse_labels(raw: Optional[list[str]], count: int) -> list[Optional[str]]:
    if not raw:
        return [None] * count
    if len(raw) != count:
        raise ValueError(
            f"--labels count ({len(raw)}) must match model count ({count})."
        )
    return list(raw)


def save_json(path: Path, sessions: list[ModelSession], seed: int) -> None:
    payload = {
        "seed": int(seed),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": [session.result_dict() for session in sessions],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


class Viewer:
    def __init__(
        self,
        sessions: list[ModelSession],
        *,
        seed: int,
        speed: float,
        width: int,
        height: int,
        save_json_path: Optional[Path],
        fall_animation: bool = True,
        stop_on_first_loss: bool = True,
    ):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required for graphical watch mode.\n"
                "Install it in the project venv with: pip install pygame"
            ) from exc

        self.pygame = pygame
        pygame.init()
        pygame.display.set_caption("Tetris Learning AI - Model Watch / Race V3.4")

        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.sessions = sessions
        self.seed = int(seed)
        self.speed = max(0.05, float(speed))
        self.paused = False
        self.pause_started_at: Optional[float] = None
        self.manual_step_animation = False
        self.step_once = False
        self.running = True
        self.show_detail = True
        self.fall_animation = bool(fall_animation)
        self.last_step_time = time.perf_counter()
        self.save_json_path = save_json_path
        self.stop_on_first_loss = bool(stop_on_first_loss)
        self.race_finished = False
        self.race_losers: list[str] = []
        self.race_winners: list[str] = []
        self.control_rects: dict[str, object] = {}

        self.font_small = pygame.font.Font(None, 20)
        self.font = pygame.font.Font(None, 23)
        self.font_med = pygame.font.Font(None, 27)
        self.font_big = pygame.font.Font(None, 34)

    def _reset_all(self, seed: int) -> None:
        self.seed = int(seed)
        for session in self.sessions:
            session.reset(self.seed)
        self.paused = False
        self.pause_started_at = None
        self.manual_step_animation = False
        self.step_once = False
        self.race_finished = False
        self.race_losers = []
        self.race_winners = []
        self.last_step_time = time.perf_counter()

    def _visual_now(self) -> float:
        """
        Time used by reconstructed piece animation.

        Normal pause freezes animation. A manual Right-arrow step is special:
        the viewer remains logically paused, but exactly one piece animation is
        allowed to advance to completion.
        """
        if self.manual_step_animation:
            return time.perf_counter()

        if self.paused and self.pause_started_at is not None:
            return self.pause_started_at

        return time.perf_counter()

    def _resume_visual_clock_for_manual_step(self) -> None:
        """
        Continue a piece that was frozen by Pause without resuming autoplay.
        """
        now = time.perf_counter()

        if self.pause_started_at is not None:
            paused_for = now - self.pause_started_at
            for session in self.sessions:
                drop = session.visual_drop
                if (
                    drop is not None
                    and drop.progress(self.pause_started_at) < 1.0
                ):
                    drop.started_at += paused_for

        self.pause_started_at = now
        self.manual_step_animation = True

    def _set_paused(self, value: bool) -> None:
        value = bool(value)
        if value == self.paused:
            return

        now = time.perf_counter()

        if value:
            self.paused = True
            self.manual_step_animation = False
            self.pause_started_at = now
            return

        # Resume: move animation start timestamps forward by the exact amount
        # of time spent paused so falling pieces do not "teleport" on resume.
        if self.pause_started_at is not None:
            paused_for = now - self.pause_started_at
            for session in self.sessions:
                drop = session.visual_drop
                if drop is not None and drop.progress(self.pause_started_at) < 1.0:
                    drop.started_at += paused_for

        self.paused = False
        self.manual_step_animation = False
        self.pause_started_at = None
        self.last_step_time = now

    def _race_has_first_loss(self) -> bool:
        if len(self.sessions) <= 1 or not self.stop_on_first_loss:
            return False
        return any(session.game_over for session in self.sessions)

    def _finalize_race_if_needed(self) -> bool:
        """
        First-top-out race semantics:
        - single model: continue until that model tops out;
        - 2+ models: as soon as any model tops out, stop the race;
        - simultaneous top-outs are a tie among the top-out models;
        - every still-alive model at that exact moment is a winner/survivor.
        """
        if self.race_finished:
            return True

        if not self._race_has_first_loss():
            return False

        self.race_finished = True
        self.race_losers = [
            s.policy.label for s in self.sessions if s.game_over
        ]
        self.race_winners = [
            s.policy.label for s in self.sessions if not s.game_over
        ]

        for session in self.sessions:
            if not session.game_over and not session.done:
                session.done = True
                session.done_reason = "RACE WON"

        self._set_paused(True)

        if self.save_json_path is not None:
            save_json(
                self.save_json_path,
                self.sessions,
                self.seed,
            )

        return True

    def _all_done(self) -> bool:
        return all(s.done for s in self.sessions)

    def _animations_active(self) -> bool:
        if not self.fall_animation:
            return False
        now = self._visual_now()
        for session in self.sessions:
            drop = session.visual_drop
            if drop is not None and drop.progress(now) < 1.0:
                return True
        return False

    def _step_round(self, *, animate: bool = True) -> None:
        for session in self.sessions:
            if not session.done:
                session.step()
                if session.visual_drop is not None:
                    if animate and self.fall_animation:
                        # V3 has two visual phases (adjust + fall), so give it
                        # slightly more readable time than the V2 straight drop.
                        session.visual_drop.duration = max(
                            0.12,
                            min(1.15, 2.20 / max(self.speed, 0.05)),
                        )
                        session.visual_drop.started_at = time.perf_counter()
                    else:
                        # Single-step while paused means "advance exactly one
                        # completed placement", not start a frozen animation.
                        session.visual_drop.duration = 0.0
                        session.visual_drop.started_at = time.perf_counter()

        if self._finalize_race_if_needed():
            return

        if self._all_done():
            self._set_paused(True)
            if self.save_json_path is not None:
                save_json(
                    self.save_json_path,
                    self.sessions,
                    self.seed,
                )

    def _save_screenshot(self) -> Path:
        folder = PROJECT_ROOT / "artifacts" / "screenshots"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = folder / f"watch_models_seed{self.seed}_{stamp}.png"
        self.pygame.image.save(self.screen, str(path))
        return path

    def _finish_active_animations(self) -> bool:
        """
        Immediately reveal the committed landing for any visible placement.
        """
        if not self.fall_animation:
            return False

        now = self._visual_now()
        finished_any = False

        for session in self.sessions:
            drop = session.visual_drop
            if drop is not None and drop.progress(now) < 1.0:
                drop.duration = 0.0
                drop.started_at = now
                finished_any = True

        if finished_any:
            self.manual_step_animation = False
            self.paused = True
            self.pause_started_at = time.perf_counter()

        return finished_any

    def _start_manual_step(self) -> None:
        """
        Run exactly one placement animation while remaining in manual-step mode.

        Important UI invariant:
        - While the selected piece is falling, HOLD/NEXT are shown from the
          pre-lock visual snapshot.
        - After it locks, the newly active piece appears at the center spawn.
          NEXT then shifts exactly once (or according to a real Hold consume).
        """
        # If a previous piece is currently frozen mid-animation, Right means
        # "continue this one piece to completion", not "skip to another piece".
        if self._animations_active():
            self._resume_visual_clock_for_manual_step()
            return

        if self._all_done():
            return

        # Remain logically paused, but permit this one animation to run.
        if not self.paused:
            self._set_paused(True)

        self.manual_step_animation = True
        self.pause_started_at = time.perf_counter()
        self._step_round(animate=True)
        self.last_step_time = time.perf_counter()

    def _run_control(self, action: str) -> None:
        pygame = self.pygame

        if action == "pause":
            if self.manual_step_animation:
                # Freeze the one-piece animation exactly where it is.
                self.manual_step_animation = False
                self.paused = True
                self.pause_started_at = time.perf_counter()
            else:
                self._set_paused(not self.paused)

        elif action == "step":
            # Repeated Right while the one-piece animation is still moving:
            # finish/reveal the current piece immediately. The next Right then
            # starts the next piece. This keeps stepping responsive.
            if self.manual_step_animation and self._animations_active():
                self._finish_active_animations()
            else:
                self._start_manual_step()

        elif action == "restart":
            self._reset_all(self.seed)
        elif action == "next":
            self._reset_all(self.seed + 1)
        elif action == "slower":
            self.speed = max(0.05, self.speed / 1.5)
        elif action == "faster":
            self.speed = min(240.0, self.speed * 1.5)
        elif action == "detail":
            self.show_detail = not self.show_detail
        elif action == "shot":
            path = self._save_screenshot()
            print(f"Screenshot: {path}")
        elif action == "quit":
            self.running = False

    def _handle_events(self) -> None:
        pygame = self.pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(
                    (event.w, event.h),
                    pygame.RESIZABLE,
                )
                continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for action, rect in self.control_rects.items():
                    if rect.collidepoint(event.pos):
                        self._run_control(action)
                        break
                continue

            if event.type != pygame.KEYDOWN:
                continue

            key = event.key
            if key in {pygame.K_ESCAPE, pygame.K_q}:
                self._run_control("quit")
            elif key == pygame.K_SPACE:
                self._run_control("pause")
            elif key == pygame.K_RIGHT:
                self._run_control("step")
            elif key == pygame.K_r:
                self._run_control("restart")
            elif key == pygame.K_n:
                self._run_control("next")
            elif key == pygame.K_d:
                self._run_control("detail")
            elif key in {pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS}:
                self._run_control("faster")
            elif key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
                self._run_control("slower")
            elif pygame.K_1 <= key <= pygame.K_9:
                digit = key - pygame.K_0
                self.speed = SPEED_PRESETS[int(digit)]
            elif key == pygame.K_s:
                self._run_control("shot")

    def _maybe_step(self) -> None:
        if self._all_done():
            return

        # Manual Right-step: let exactly one reconstructed movement animation
        # run even though autoplay remains paused.
        if self.manual_step_animation:
            if self._animations_active():
                return

            # The one piece has reached its landing and the committed post-state
            # is now shown. Freeze here until the user presses Right again.
            self.manual_step_animation = False
            self.paused = True
            self.pause_started_at = time.perf_counter()
            self.last_step_time = self.pause_started_at
            return

        if self.paused:
            return

        # Automatic playback waits for the current visual placement to finish.
        if self._animations_active():
            return

        now = time.perf_counter()
        interval = 1.0 / self.speed
        if now - self.last_step_time >= interval:
            self.last_step_time = now
            self._step_round()

    def _text(
        self,
        surface,
        text: str,
        x: int,
        y: int,
        *,
        font=None,
        color=TEXT,
    ) -> int:
        font = font or self.font
        img = font.render(str(text), True, color)
        surface.blit(img, (x, y))
        return img.get_height()

    def _draw_block(
        self,
        surface,
        rect,
        color,
        *,
        ghost: bool = False,
    ) -> None:
        pygame = self.pygame

        if ghost:
            ghost_fill = tuple(max(25, int(c * 0.24)) for c in color)
            pygame.draw.rect(surface, ghost_fill, rect.inflate(-4, -4))
            pygame.draw.rect(surface, color, rect.inflate(-4, -4), width=1)
            return

        inner = rect.inflate(-2, -2)
        pygame.draw.rect(surface, darken(color, 0.55), inner, border_radius=2)

        face = pygame.Rect(
            inner.x + 2,
            inner.y + 2,
            max(1, inner.width - 4),
            max(1, inner.height - 4),
        )
        pygame.draw.rect(surface, color, face, border_radius=2)

        if rect.width >= 8 and rect.height >= 8:
            pygame.draw.line(
                surface,
                brighten(color, 55),
                (face.left + 1, face.top + 1),
                (face.right - 1, face.top + 1),
                width=1,
            )
            pygame.draw.line(
                surface,
                brighten(color, 28),
                (face.left + 1, face.top + 1),
                (face.left + 1, face.bottom - 1),
                width=1,
            )

    def _draw_board(
        self,
        surface,
        board_ids: np.ndarray,
        x: int,
        y: int,
        cell: int,
        *,
        overlay_piece: Optional[str] = None,
        overlay_shape: Optional[np.ndarray] = None,
        overlay_x: int = 0,
        overlay_y: int = 0,
        ghost_piece: Optional[str] = None,
        ghost_shape: Optional[np.ndarray] = None,
        ghost_x: Optional[int] = None,
        ghost_y: Optional[int] = None,
    ) -> None:
        pygame = self.pygame
        board_ids = np.asarray(board_ids).reshape(20, 10)

        pygame.draw.rect(
            surface,
            PANEL_BORDER,
            (x - 2, y - 2, 10 * cell + 4, 20 * cell + 4),
            width=2,
        )

        for row in range(20):
            for col in range(10):
                rect = pygame.Rect(
                    x + col * cell,
                    y + row * cell,
                    cell,
                    cell,
                )
                piece_id = int(board_ids[row, col])

                if piece_id >= 2:
                    name = PIECE_ID_TO_NAME.get(piece_id)
                    self._draw_block(
                        surface,
                        rect,
                        piece_color(name),
                    )
                elif piece_id:
                    # Fallback for a nonstandard occupied board encoding.
                    self._draw_block(
                        surface,
                        rect,
                        (170, 176, 188),
                    )
                else:
                    pygame.draw.rect(
                        surface,
                        EMPTY,
                        rect.inflate(-1, -1),
                    )

                pygame.draw.rect(surface, GRID, rect, width=1)

        if overlay_piece is None or overlay_shape is None:
            return

        color = piece_color(overlay_piece)
        shape = np.asarray(overlay_shape, dtype=np.uint8)

        # Exact final-placement ghost, independent of the current rotation/x
        # being shown during the human-like adjustment phase.
        if (
            ghost_y is not None
            and ghost_shape is not None
            and ghost_x is not None
        ):
            g_piece = ghost_piece or overlay_piece
            g_color = piece_color(g_piece)
            g_shape = np.asarray(ghost_shape, dtype=np.uint8)
            for sy in range(g_shape.shape[0]):
                for sx in range(g_shape.shape[1]):
                    if not g_shape[sy, sx]:
                        continue
                    bx = int(ghost_x) + sx
                    by = int(ghost_y) + sy
                    if not (0 <= bx < 10 and 0 <= by < 20):
                        continue
                    rect = pygame.Rect(
                        x + bx * cell,
                        y + by * cell,
                        cell,
                        cell,
                    )
                    self._draw_block(
                        surface,
                        rect,
                        g_color,
                        ghost=True,
                    )

        # Falling active tetromino.
        for sy in range(shape.shape[0]):
            for sx in range(shape.shape[1]):
                if not shape[sy, sx]:
                    continue

                bx = overlay_x + sx
                by = overlay_y + sy
                if not (0 <= bx < 10 and 0 <= by < 20):
                    continue

                rect = pygame.Rect(
                    x + bx * cell,
                    y + by * cell,
                    cell,
                    cell,
                )
                self._draw_block(surface, rect, color)

    def _draw_mini_piece(
        self,
        surface,
        piece: Optional[str],
        rect,
        *,
        label: Optional[str] = None,
    ) -> None:
        pygame = self.pygame

        pygame.draw.rect(
            surface,
            (22, 25, 31),
            rect,
            border_radius=5,
        )
        pygame.draw.rect(
            surface,
            PANEL_BORDER,
            rect,
            width=1,
            border_radius=5,
        )

        if label:
            self._text(
                surface,
                label,
                rect.x + 5,
                rect.y + 3,
                font=self.font_small,
                color=MUTED,
            )

        if not piece:
            self._text(
                surface,
                "-",
                rect.centerx - 3,
                rect.centery - 9,
                font=self.font_med,
                color=MUTED,
            )
            return

        try:
            shape = dict(get_rotations(str(piece)))[0]
        except Exception:
            return

        shape = np.asarray(shape, dtype=np.uint8)
        label_h = 18 if label else 4
        available_h = max(8, rect.height - label_h - 6)
        available_w = max(8, rect.width - 10)
        mini_cell = max(
            3,
            min(
                available_w // max(1, shape.shape[1]),
                available_h // max(1, shape.shape[0]),
                14,
            ),
        )

        draw_w = shape.shape[1] * mini_cell
        draw_h = shape.shape[0] * mini_cell
        ox = rect.centerx - draw_w // 2
        oy = rect.y + label_h + max(0, (available_h - draw_h) // 2)

        color = piece_color(str(piece))
        for sy in range(shape.shape[0]):
            for sx in range(shape.shape[1]):
                if not shape[sy, sx]:
                    continue
                block = pygame.Rect(
                    ox + sx * mini_cell,
                    oy + sy * mini_cell,
                    mini_cell,
                    mini_cell,
                )
                self._draw_block(surface, block, color)

    def _draw_hold_next(
        self,
        surface,
        *,
        hold_piece: Optional[str],
        next_pieces: tuple[str, ...],
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        pygame = self.pygame

        preview_w = max(64, min(92, width))
        hold_h = 72
        gap = 6

        hold_rect = pygame.Rect(x, y, preview_w, hold_h)
        self._draw_mini_piece(
            surface,
            hold_piece,
            hold_rect,
            label="HOLD",
        )

        next_y = hold_rect.bottom + gap
        self._text(
            surface,
            "NEXT",
            x + 4,
            next_y,
            font=self.font_small,
            color=MUTED,
        )
        next_y += 18

        available = max(0, height - (next_y - y))
        count = min(4, len(next_pieces))
        if count == 0:
            return

        box_h = max(38, min(62, (available - gap * (count - 1)) // count))
        for i, piece in enumerate(next_pieces[:count]):
            box = pygame.Rect(
                x,
                next_y + i * (box_h + gap),
                preview_w,
                box_h,
            )
            self._draw_mini_piece(surface, piece, box)

    def _draw_session(
        self,
        surface,
        session: ModelSession,
        rect,
        index: int,
        leader: Optional[ModelSession],
    ) -> None:
        pygame = self.pygame
        x, y, w, h = rect
        color = PALETTE[index % len(PALETTE)]

        pygame.draw.rect(surface, PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER, rect, width=1, border_radius=8)

        title = session.policy.label
        if leader is session and len(self.sessions) > 1:
            title = "★ " + title

        self._text(
            surface,
            title,
            x + 14,
            y + 10,
            font=self.font_med,
            color=color,
        )

        max_board_h = max(120, h - 74)
        cell = max(
            5,
            min(
                max_board_h // 20,
                max(5, int(w * 0.40) // 10),
                30,
            ),
        )

        board_x = x + 14
        board_y = y + 44

        now = self._visual_now()
        drop = session.visual_drop
        animating = (
            self.fall_animation
            and drop is not None
            and drop.progress(now) < 1.0
        )

        if animating:
            display_board = drop.board_ids
            overlay_piece = drop.piece
            overlay_shape, overlay_x, overlay_y, _ = drop.pose(now)
            ghost_piece = drop.piece
            ghost_shape = drop.target_shape
            ghost_x = drop.target_x
            ghost_y = drop.landing_y
            display_hold = drop.hold_piece
            display_next = drop.next_pieces
        else:
            display_board = session.locked_board_ids()

            # Show the *current* active tetromino at spawn whenever we are
            # between placements. This is the key NEXT-queue clarity fix:
            # after a lock, previous NEXT[0] becomes Current and is visibly
            # present on the board; NEXT therefore no longer looks as if it
            # skipped a piece.
            overlay_piece = session.state.current_piece
            try:
                overlay_shape = dict(
                    get_rotations(overlay_piece)
                )[0]
                overlay_shape = np.asarray(
                    overlay_shape,
                    dtype=np.uint8,
                )
                overlay_x = max(
                    0,
                    (10 - int(overlay_shape.shape[1])) // 2,
                )
                overlay_y = 0
            except Exception:
                overlay_shape = None
                overlay_x = 0
                overlay_y = 0

            ghost_piece = None
            ghost_shape = None
            ghost_x = None
            ghost_y = None
            display_hold = session.state.hold_piece
            display_next = tuple(session.state.next_pieces)

        self._draw_board(
            surface,
            display_board,
            board_x,
            board_y,
            cell,
            overlay_piece=overlay_piece,
            overlay_shape=overlay_shape,
            overlay_x=overlay_x,
            overlay_y=overlay_y,
            ghost_piece=ghost_piece,
            ghost_shape=ghost_shape,
            ghost_x=ghost_x,
            ghost_y=ghost_y,
        )

        preview_w = 86 if w >= 650 else 68
        preview_x = x + w - preview_w - 12
        preview_y = board_y
        preview_h = max(150, h - (preview_y - y) - 18)
        self._draw_hold_next(
            surface,
            hold_piece=display_hold,
            next_pieces=display_next,
            x=preview_x,
            y=preview_y,
            width=preview_w,
            height=preview_h,
        )

        stats_x = board_x + cell * 10 + 18
        stats_y = board_y
        stats_right = preview_x - 8
        line_h = 22

        status_color = BAD if session.game_over else (WARN if session.done else GOOD)
        status = session.done_reason if session.done else "RUNNING"

        rows = [
            ("Status", status, status_color),
            ("Pieces", str(session.stats.pieces), TEXT),
            ("Lines", str(session.stats.lines), TEXT),
            ("Tetris", str(session.stats.tetrises), TEXT),
            ("Value", f"{session.stats.value:,}", TEXT),
            ("Height", f"{session.stats.current_height} / max {session.stats.max_height}", TEXT),
            ("Holes", f"{session.stats.holes} / max {session.stats.max_holes}", TEXT),
            ("Avg H", f"{session.stats.avg_height:.2f}", TEXT),
            (
                "Q switch",
                f"{session.stats.interventions} ({session.stats.switch_rate * 100:.1f}%)",
                TEXT,
            ),
            ("Policy", session.policy.gate_short, MUTED),
        ]

        for label, value, value_color in rows:
            self._text(surface, f"{label}:", stats_x, stats_y, color=MUTED)
            self._text(surface, value, stats_x + 88, stats_y, color=value_color)
            stats_y += line_h

        if self.show_detail and stats_y + 90 < y + h:
            stats_y += 4
            self._text(
                surface,
                f"Piece {session.last.piece}  {session.last.action_text}",
                stats_x,
                stats_y,
                color=TEXT,
            )
            stats_y += line_h
            source_color = (
                TEACHER_COLOR
                if session.last.chosen_index == 0
                else GOOD
            )
            self._text(
                surface,
                f"{session.last.source}  conf={session.last.confidence:.3f}",
                stats_x,
                stats_y,
                color=source_color,
            )
            stats_y += line_h

            if session.last.q_values:
                q_str = "  ".join(
                    f"q{i + 1}={q:.3f}"
                    for i, q in enumerate(session.last.q_values)
                )
                self._text(
                    surface,
                    q_str[:58],
                    stats_x,
                    stats_y,
                    font=self.font_small,
                    color=MUTED,
                )
                stats_y += 19

            current_piece = (
                drop.piece
                if animating and drop is not None
                else session.state.current_piece
            )
            self._text(
                surface,
                f"Current: {current_piece}",
                stats_x,
                stats_y,
                font=self.font_small,
                color=piece_color(current_piece),
            )

        if session.last_exception:
            self._text(
                surface,
                session.last_exception[:72],
                x + 14,
                y + h - 25,
                font=self.font_small,
                color=BAD,
            )

    def _draw_controls(
        self,
        surface,
        *,
        y: int,
        width: int,
        height: int,
    ) -> None:
        pygame = self.pygame

        bar = pygame.Rect(8, y, max(100, width - 16), height)
        pygame.draw.rect(surface, (22, 25, 31), bar, border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER, bar, width=1, border_radius=8)

        controls = [
            ("pause", "[Space] Resume" if self.paused else "[Space] Pause"),
            ("step", "[Right] One piece ▶"),
            ("restart", "[R] Restart"),
            ("next", "[N] Next seed"),
            ("slower", "[-] Slower"),
            ("faster", "[+] Faster"),
            ("detail", "[D] Detail"),
            ("shot", "[S] Screenshot"),
            ("quit", "[Esc] Quit"),
        ]

        self.control_rects = {}
        x = bar.x + 8
        cy = bar.y + 7
        row_h = 28

        mouse_pos = pygame.mouse.get_pos()

        for action, label in controls:
            label_img = self.font_small.render(label, True, TEXT)
            chip_w = label_img.get_width() + 18

            if x + chip_w > bar.right - 8:
                x = bar.x + 8
                cy += row_h

            rect = pygame.Rect(x, cy, chip_w, 23)
            hovered = rect.collidepoint(mouse_pos)
            fill = (53, 59, 70) if hovered else (35, 39, 47)

            if action == "pause" and self.paused:
                fill = (69, 61, 38)

            pygame.draw.rect(surface, fill, rect, border_radius=5)
            pygame.draw.rect(surface, PANEL_BORDER, rect, width=1, border_radius=5)
            surface.blit(
                label_img,
                (rect.x + 9, rect.y + 3),
            )

            self.control_rects[action] = rect
            x = rect.right + 6

        if self.manual_step_animation:
            state_name = "STEP PLAYING"
        elif self.paused:
            state_name = "PAUSED"
        else:
            state_name = "PLAYING"

        state_text = (
            f"State: {state_name}"
            f"   Speed: {self.speed:.1f} pieces/s"
            f"   Seed: {self.seed}"
        )
        self._text(
            surface,
            state_text,
            bar.x + 10,
            bar.bottom - 23,
            font=self.font_small,
            color=WARN if self.paused else GOOD,
        )

    def _draw(self) -> None:
        pygame = self.pygame
        screen = self.screen
        width, height = screen.get_size()
        screen.fill(BG)

        n = len(self.sessions)
        cols = 1 if n == 1 else (2 if n <= 4 else math.ceil(math.sqrt(n)))
        rows = math.ceil(n / cols)

        header_h = 52
        footer_h = 82
        gap = 8
        pad = 10

        usable_w = width - 2 * pad - gap * (cols - 1)
        usable_h = height - header_h - footer_h - pad - gap * (rows - 1)
        panel_w = max(260, usable_w // cols)
        panel_h = max(250, usable_h // rows)

        mode = "Single Model Watch" if n == 1 else f"{n}-Model Race"
        state = (
            "STEP"
            if self.manual_step_animation
            else ("PAUSED" if self.paused else "PLAY")
        )
        if self.race_finished and len(self.sessions) > 1:
            if self.race_winners:
                race_summary = "Winner: " + ", ".join(self.race_winners)
            else:
                race_summary = "Race ended in a simultaneous top-out"
            header_text = (
                f"{mode}   Seed {self.seed}   {race_summary}"
            )
        else:
            header_text = (
                f"{mode}   Seed {self.seed}   {state}   "
                f"{self.speed:.1f} pieces/s"
            )

        self._text(
            screen,
            header_text,
            14,
            12,
            font=self.font_big,
            color=GOOD if self.race_finished else TEXT,
        )

        active = [s for s in self.sessions if not s.done]
        candidates = active if active else self.sessions
        leader = max(candidates, key=race_rank) if candidates else None

        for i, session in enumerate(self.sessions):
            row = i // cols
            col = i % cols
            rect = pygame.Rect(
                pad + col * (panel_w + gap),
                header_h + row * (panel_h + gap),
                panel_w,
                panel_h,
            )
            self._draw_session(screen, session, rect, i, leader)

        self._draw_controls(
            screen,
            y=height - footer_h + 4,
            width=width,
            height=footer_h - 8,
        )

        pygame.display.flip()

    def run(self) -> None:
        while self.running:
            self._handle_events()
            self._maybe_step()
            self._draw()
            self.clock.tick(60)

        self.pygame.quit()


def print_results(sessions: list[ModelSession]) -> None:
    ordered = sorted(sessions, key=race_rank, reverse=True)

    print()
    print("=" * 108)
    print("FINAL RESULTS")
    print("=" * 108)
    header = (
        f"{'#':>2}  {'Model':<38} {'Pieces':>7} {'Lines':>6} "
        f"{'Tetris':>7} {'Value':>10} {'H':>3} {'Holes':>5} "
        f"{'Switch%':>8} {'Result':>10}"
    )
    print(header)
    print("-" * len(header))

    for rank, session in enumerate(ordered, 1):
        r = session.result_dict()
        label = r["label"][:38]
        print(
            f"{rank:>2}  {label:<38} "
            f"{r['pieces']:>7} {r['lines']:>6} {r['tetrises']:>7} "
            f"{r['value']:>10,} {r['max_height']:>3} {r['max_holes']:>5} "
            f"{r['switch_rate'] * 100:>7.2f}% {r['done_reason']:>10}"
        )

    print("=" * 108)


def fast_forward_sessions(
    sessions: list[ModelSession],
    *,
    start_piece: int,
    stop_on_first_loss: bool,
) -> tuple[bool, str]:
    """
    Advance without rendering until the requested piece is about to be played.

    Example:
      --start-piece 3850

    completes placements 1..3849 headlessly, clears any reconstructed visual
    drop, and returns with Current = piece 3850 ready at spawn.

    Returns:
      (reached_requested_piece, reason)
    """
    target_completed = max(0, int(start_piece) - 1)

    if target_completed == 0:
        for session in sessions:
            session.visual_drop = None
        return True, "READY"

    print()
    print(
        f"Fast-forwarding to piece {start_piece:,} "
        f"(executing {target_completed:,} placements headlessly)..."
    )

    started = time.perf_counter()
    next_report = 250
    multi = len(sessions) > 1

    while True:
        # Requested inspection point reached for every still-relevant model.
        if all(
            session.stats.pieces >= target_completed
            for session in sessions
        ):
            reason = "READY"
            reached = True
            break

        for session in sessions:
            if (
                not session.done
                and session.stats.pieces < target_completed
            ):
                session.step()

        # Do not retain a stale visual reconstruction during fast-forward.
        for session in sessions:
            session.visual_drop = None

        if (
            multi
            and stop_on_first_loss
            and any(session.game_over for session in sessions)
        ):
            reached = False
            reason = "TOP_OUT_BEFORE_START"
            break

        if all(session.done for session in sessions):
            reached = False
            reason = "ALL_DONE_BEFORE_START"
            break

        common_pieces = min(
            session.stats.pieces
            for session in sessions
        )

        if common_pieces >= next_report:
            elapsed = time.perf_counter() - started
            rate = common_pieces / max(elapsed, 1e-9)
            remaining = max(0, target_completed - common_pieces)
            eta = remaining / max(rate, 1e-9)

            status = " | ".join(
                f"{session.policy.label}: P{session.stats.pieces:,}"
                for session in sessions
            )
            print(
                f"  {status} | "
                f"{rate:.1f} placements/s | ETA {eta:.1f}s"
            )
            next_report += 250

    elapsed = time.perf_counter() - started

    # The viewer should always open on the stable post-fast-forward state:
    # no hidden animation from the last skipped placement.
    for session in sessions:
        session.visual_drop = None

    if reached:
        print(
            f"Fast-forward complete in {elapsed:.2f}s. "
            f"Piece {start_piece:,} is ready. Viewer will open PAUSED."
        )
    else:
        print(
            f"Could not reach piece {start_piece:,}: {reason} "
            f"(elapsed {elapsed:.2f}s)."
        )
        for session in sessions:
            print(
                f"  {session.policy.label}: "
                f"pieces={session.stats.pieces:,}, "
                f"status={session.done_reason or 'ACTIVE'}"
            )

    return reached, reason


def run_headless(
    sessions: list[ModelSession],
    *,
    stop_on_first_loss: bool = True,
) -> None:
    multi = len(sessions) > 1

    while not all(s.done for s in sessions):
        for session in sessions:
            if not session.done:
                session.step()

        if (
            multi
            and stop_on_first_loss
            and any(session.game_over for session in sessions)
        ):
            for session in sessions:
                if not session.game_over and not session.done:
                    session.done = True
                    session.done_reason = "RACE WON"
            break

    print_results(sessions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Watch one model until top-out, race multiple models, or fast-forward to a chosen piece for visual inspection. "
            "Use the special model spec 'teacher' for a Teacher-only baseline."
        )
    )

    parser.add_argument(
        "models",
        nargs="+",
        help=(
            "Checkpoint paths. One model = single watch; 2+ = race. "
            "Use 'teacher' for the heuristic Teacher baseline."
        ),
    )
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=50001)
    parser.add_argument(
        "--start-piece",
        type=int,
        default=1,
        help=(
            "First piece to watch. Example: --start-piece 3850 runs pieces "
            "1..3849 headlessly, then opens the viewer paused with piece 3850 "
            "ready at the center spawn. Default 1."
        ),
    )
    parser.add_argument(
        "--max-pieces",
        type=int,
        default=0,
        help=(
            "Maximum pieces per model. 0 (default) = unlimited, play until "
            "game over/top-out."
        ),
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--continue-after-loss",
        action="store_true",
        help=(
            "For multi-model races, keep surviving models playing after the "
            "first top-out. Default behavior stops the race immediately when "
            "the first model loses."
        ),
    )
    parser.add_argument("--speed", type=float, default=6.0)
    parser.add_argument(
        "--no-fall-animation",
        action="store_true",
        help="Disable visual falling animation and show committed boards immediately.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gate", type=float, default=None)
    parser.add_argument(
        "--gate-semantics",
        choices=["auto", "normalized_q_margin", "raw_q_gap"],
        default="auto",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument(
        "--allow-protected-seeds",
        action="store_true",
        help="Allow permanent final benchmark seeds 6..20.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.seed in PROTECTED_FINAL_SEEDS and not args.allow_protected_seeds:
        raise SystemExit(
            "Seed 6..20 is protected for final benchmark/report use. "
            "Choose another seed, or explicitly pass --allow-protected-seeds."
        )

    if args.max_pieces < 0:
        raise SystemExit("--max-pieces must be >= 0 (0 = unlimited).")
    if args.start_piece < 1:
        raise SystemExit("--start-piece must be >= 1.")
    if args.max_pieces > 0 and args.start_piece > args.max_pieces:
        raise SystemExit(
            "--start-piece cannot be greater than a finite --max-pieces cap."
        )
    if args.headless and args.start_piece != 1:
        raise SystemExit(
            "--start-piece is a graphical inspection option. "
            "Do not combine it with --headless."
        )
    if not 1 <= args.top_k <= 4:
        raise SystemExit("--top-k must be between 1 and 4.")
    if args.speed <= 0:
        raise SystemExit("--speed must be > 0.")
    if args.gate is not None and args.gate < 0:
        raise SystemExit("--gate must be >= 0.")

    device = choose_device(args.device)
    labels = parse_labels(args.labels, len(args.models))

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Models: {len(args.models)}")
    print()

    policies = []
    for spec, label in zip(args.models, labels):
        policy = load_policy(
            spec,
            label=label,
            device=device,
            gate_override=args.gate,
            semantics_override=args.gate_semantics,
        )
        policies.append(policy)
        print(
            f"Loaded: {policy.label} | "
            f"{policy.gate_short} | "
            f"env_steps={short_steps(policy.env_steps)}"
        )

    teacher = HeuristicTeacherV2()
    sessions = [
        ModelSession(
            policy,
            seed=args.seed,
            max_pieces=args.max_pieces,
            top_k=args.top_k,
            device=device,
            teacher=teacher,
        )
        for policy in policies
    ]

    try:
        if args.headless:
            run_headless(
                sessions,
                stop_on_first_loss=not args.continue_after_loss,
            )
            if args.save_json is not None:
                path = args.save_json
                if not path.is_absolute():
                    path = PROJECT_ROOT / path
                save_json(path, sessions, args.seed)
                print(f"Saved JSON: {path}")
            return

        fast_forward_reached = True
        if args.start_piece > 1:
            fast_forward_reached, fast_forward_reason = fast_forward_sessions(
                sessions,
                start_piece=args.start_piece,
                stop_on_first_loss=not args.continue_after_loss,
            )

            if not fast_forward_reached:
                print()
                print(
                    "Viewer will open at the latest reachable state instead. "
                    "Choose a smaller --start-piece to inspect before the loss."
                )

        count = len(sessions)
        width = args.width
        height = args.height

        if width is None:
            width = 1120 if count == 1 else (1600 if count <= 4 else 1800)
        if height is None:
            height = 860 if count <= 2 else 1000

        save_path = args.save_json
        if save_path is not None and not save_path.is_absolute():
            save_path = PROJECT_ROOT / save_path

        viewer = Viewer(
            sessions,
            seed=args.seed,
            speed=args.speed,
            width=width,
            height=height,
            save_json_path=save_path,
            fall_animation=not args.no_fall_animation,
            stop_on_first_loss=not args.continue_after_loss,
        )

        if args.start_piece > 1:
            viewer._set_paused(True)

            # If a model already topped out before the requested point, make
            # the race result visible immediately rather than waiting for a
            # nonexistent next placement.
            if any(session.game_over for session in sessions):
                viewer._finalize_race_if_needed()

        viewer.run()

        print_results(sessions)

        if save_path is not None:
            save_json(save_path, sessions, viewer.seed)
            print(f"Saved JSON: {save_path}")

    finally:
        for session in sessions:
            try:
                session.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
