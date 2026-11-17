"""Footstep planner: where each foot should be, relative to the hips, at each moment of the step cycle.

Like a game engine's foot placement: a walk cycle says where the feet go, and IK (leg_kinematics.py) turns
that into joint angles. Each foot is on the ground for ``stance_fraction`` of the cycle, the other foot half a
cycle later, so one foot is always down. A planted foot stays put on the ground, so in the robot's own frame it
slides back under the hips (and turns, when turning). The swinging foot lifts, swings through on a smooth arc,
and lands half a step ahead of the hips (behind them walking backward), heel first, while the other foot is
behind: left in front, then right in front. The feet stay ``width_m`` apart (about hip width).

Catch step: given the hips' velocity, a swinging foot lands further the way the body is moving faster than
commanded (the capture point), so the plan itself recovers from a stumble.

Weight shift: like a person, the hips sway over the standing foot each step (``hip_sway_m``), and hips that drift
off the feet are pulled back over them. The planter gives this as ``hips_offset``: where the hips should be,
relative to where they are; the foot targets are measured from there, so following them moves the hips.

Frames: x forward, y left, z up, origin at the hips (pelvis), level with the ground. Feet are [left, right].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GaitPlan:
    period_s: float = 0.8  # one left step and one right step
    stance_fraction: float = 0.6  # share of the cycle each foot is on the ground
    step_height_m: float = 0.08  # how high the swinging foot lifts
    width_m: float = 0.20  # feet apart, centre to centre: about the R1's hip width, a narrow human-like track
    foot_x_m: float = 0.0  # where the feet are under the hips when standing (forward of the pelvis origin)
    foot_z_m: float = -0.655  # ankle height relative to the pelvis while walking (knees softly bent; room to reach)
    stand_foot_z_m: float = -0.683  # ankle height relative to the pelvis standing straight (knees about 10 deg)
    heel_tilt_rad: float = 0.17  # toes up by about 10 degrees when the foot lands
    moving_threshold: float = 0.05
    max_step_m: float = 0.30  # longest step (how far the hips travel while a foot is down); faster = step more often
    min_period_s: float = 0.5  # quickest step cycle
    arm_swing_rad: float = 0.45  # shoulder swing each way when walking (about 26 deg; each arm with its own leg)
    elbow_swing_rad: float = 0.28  # extra elbow bend with the arm fully forward (about 16 deg)
    full_swing_speed: float = 1.0  # m/s at which the arms swing fully; slower walking swings less
    hip_sway_m: float = 0.03  # how far the hips shift sideways over the standing foot while walking
    catch_gain_s: float = 0.267  # catch step: landing moves by this times (actual - commanded) velocity; sqrt(h/g)
    max_catch_step_m: float = 0.12  # the largest catch-step correction, each way
    max_com_shift_m: float = 0.06  # the largest hip shift that keeps the centre of mass over the feet
    stumble_speed: float = 0.2  # m/s off the command that counts as pushed or stumbling: start stepping


def cycle_period(plan: GaitPlan, command: torch.Tensor) -> torch.Tensor:
    """Step cycle time (N,) for ``command``: the normal cycle, shorter when the steps would get too long.

    Callers advance the phase by dt / cycle_period each step.
    """
    speed = torch.linalg.norm(command[:, :2], dim=1)
    needed = plan.max_step_m / (speed * plan.stance_fraction).clamp(min=1e-6)
    return needed.clamp(plan.min_period_s, plan.period_s)


@dataclass
class FeetPlan:
    position: torch.Tensor  # (N, 2, 3) ankle positions relative to the hips
    rotation: torch.Tensor  # (N, 2, 3, 3) foot orientations relative to the hips
    stance: torch.Tensor  # (N, 2) bool: the foot should be on the ground
    hips_offset: torch.Tensor | None = None  # (N, 2) hips frame: where the hips should be, relative to now


def _yaw_pitch_matrix(yaw: torch.Tensor, toe_up: torch.Tensor) -> torch.Tensor:
    """Rz(yaw) @ Ry(-toe_up): the foot turned by ``yaw`` with its toes tipped up by ``toe_up``."""
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(-toe_up), torch.sin(-toe_up)
    zero, one = torch.zeros_like(yaw), torch.ones_like(yaw)
    rz = torch.stack([torch.stack([cy, -sy, zero], -1), torch.stack([sy, cy, zero], -1),
                      torch.stack([zero, zero, one], -1)], -2)
    ry = torch.stack([torch.stack([cp, zero, sp], -1), torch.stack([zero, one, zero], -1),
                      torch.stack([-sp, zero, cp], -1)], -2)
    return rz @ ry


def step_walk_in(
    walk_in: torch.Tensor, moving: torch.Tensor, dt: float, walk_in_s: float, stumbling: torch.Tensor | None = None
) -> torch.Tensor:
    """Advance how much each robot is walking (0 = standing, 1 = walking): up over ``walk_in_s`` while it is told
    to move, and back down at the same rate after a stop, so stopping eases out like starting eases in. A
    ``stumbling`` robot (see needs_to_step) starts stepping within a quarter of a second, to catch itself."""
    rise = torch.full_like(walk_in, dt / walk_in_s)
    if stumbling is not None:
        rise = torch.where(stumbling, torch.full_like(walk_in, dt / STUMBLE_STEP_IN_S), rise)
        moving = moving | stumbling
    change = torch.where(moving, rise, torch.full_like(walk_in, -dt / walk_in_s))
    return (walk_in + change).clamp(0.0, 1.0)


STUMBLE_STEP_IN_S = 0.25


def needs_to_step(plan: GaitPlan, command: torch.Tensor, hips_velocity: torch.Tensor) -> torch.Tensor:
    """(N,) bool: the hips move more than ``stumble_speed`` off the commanded velocity (pushed or stumbling).

    ``command`` is (N, 3) in the robot's frame and ``hips_velocity`` (N, 2) in the same frame."""
    return torch.linalg.norm(hips_velocity - command[:, :2], dim=1) > plan.stumble_speed


def hip_sway(plan: GaitPlan, phase: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    """Sideways hip shift (N,), +y = left: over the left foot while the right foot swings, and the other way.

    The right foot swings at phase 0.1-0.5 and the left at 0.6-1.0; the shift crosses zero in the middle of each
    double support (phase 0.05 and 0.55), so the weight moves across while both feet are down.
    """
    return plan.hip_sway_m * amount * torch.sin(2 * torch.pi * (phase - 0.05))


def plan_feet(
    plan: GaitPlan, command: torch.Tensor, phase: torch.Tensor, walk_in: torch.Tensor | None = None
) -> FeetPlan:
    """Foot targets for ``command`` (N, 3: forward m/s, left m/s, turn rad/s) at ``phase`` (N,) of the cycle.

    ``walk_in`` (N,): 0 = standing straight, feet side by side; 1 = walking fully. Raise it over the first step
    so the robot starts walking from standing smoothly (step length, foot lift, heel tilt and hip height blend).
    """
    device = command.device
    stance_time = cycle_period(plan, command) * plan.stance_fraction  # (N,)
    moving = (torch.linalg.norm(command[:, :2], dim=1) > plan.moving_threshold) | (
        torch.abs(command[:, 2]) > plan.moving_threshold
    )
    amount = moving.float() * (torch.ones_like(phase) if walk_in is None else walk_in.clamp(0.0, 1.0))
    travel = command[:, :2] * (stance_time * amount).unsqueeze(1)  # (N, 2): hips move this far per stance
    turn = command[:, 2] * stance_time * amount  # (N,)

    foot_phase = torch.stack([phase, torch.remainder(phase + 0.5, 1.0)], dim=1)  # (N, 2): left, right
    on_ground = foot_phase < plan.stance_fraction
    stance_progress = foot_phase / plan.stance_fraction
    swing_progress = (foot_phase - plan.stance_fraction) / (1 - plan.stance_fraction)
    smooth = (1 - torch.cos(torch.pi * swing_progress.clamp(0, 1))) / 2
    # where along the step each foot is: +0.5 = half a step ahead of the hips, -0.5 = half a step behind
    along = torch.where(on_ground, 0.5 - stance_progress, -0.5 + smooth)

    side = torch.tensor([0.5, -0.5], device=device) * plan.width_m
    nominal = torch.stack([torch.full((2,), plan.foot_x_m, device=device), side], dim=1)  # (2, 2)
    yaw = along * turn.unsqueeze(1)  # (N, 2)
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    x = cos * nominal[:, 0] - sin * nominal[:, 1] + along * travel[:, :1]
    y = sin * nominal[:, 0] + cos * nominal[:, 1] + along * travel[:, 1:]
    lift = torch.where(on_ground, torch.zeros_like(smooth), torch.sin(torch.pi * swing_progress.clamp(0, 1)))
    height = plan.stand_foot_z_m + (plan.foot_z_m - plan.stand_foot_z_m) * amount
    z = height.unsqueeze(1) + plan.step_height_m * lift * amount.unsqueeze(1)

    # heel first: toes tip up over the last 40% of the swing and lay flat over the first 15% of the stance
    tilt = torch.where(
        on_ground,
        (1 - stance_progress / 0.15).clamp(0, 1),
        ((swing_progress - 0.6) / 0.4).clamp(0, 1),
    ) * plan.heel_tilt_rad * amount.unsqueeze(1)
    sway = hip_sway(plan, phase, amount)
    return FeetPlan(
        position=torch.stack([x, y - sway.unsqueeze(1), z], dim=-1),
        rotation=_yaw_pitch_matrix(yaw, tilt),
        stance=on_ground | (amount <= 0).unsqueeze(1),
        hips_offset=torch.stack([torch.zeros_like(sway), sway], dim=1),
    )


def _rotate(xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate (..., 2) vectors by ``yaw`` (broadcast over the leading dimensions)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]], -1)


class FootPlanter:
    """Foot planting like a game engine: a planted foot stays locked to its spot on the ground.

    Each step: the planted feet keep their place in the world; a swinging foot travels from where it lifted off
    to its landing spot, half a step ahead of where the hips will be when it lands (behind, walking backward),
    turned with the walk, half of ``width_m`` from the middle. Returns the foot targets relative to the hips for the
    IK, in the same form as ``plan_feet``. Batched over ``count`` robots.
    """

    def __init__(self, plan: GaitPlan, count: int, device: torch.device | str = "cpu") -> None:
        self.plan = plan
        side = torch.tensor([0.5, -0.5], device=device) * plan.width_m
        self._nominal = torch.stack([torch.full((2,), plan.foot_x_m, device=device), side], dim=1)  # (2, 2)
        self.feet_world_xy = torch.zeros(count, 2, 2, device=device)
        self.feet_world_yaw = torch.zeros(count, 2, device=device)
        self._liftoff_xy = torch.zeros_like(self.feet_world_xy)
        self._liftoff_yaw = torch.zeros_like(self.feet_world_yaw)
        self._was_stance = torch.ones(count, 2, dtype=torch.bool, device=device)
        self._placed = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids: torch.Tensor | None = None) -> None:
        """Put these robots' feet back under their hips at the next step (after a reset in training)."""
        self._placed[slice(None) if ids is None else ids] = False

    def step(
        self, hips_xy: torch.Tensor, hips_yaw: torch.Tensor, command: torch.Tensor, phase: torch.Tensor,
        walk_in: torch.Tensor, hips_velocity: torch.Tensor | None = None, com_shift: torch.Tensor | None = None,
    ) -> FeetPlan:
        """``hips_velocity`` (N, 2, world): when given, a swinging foot lands where the body is tipping (catch step).
        ``com_shift`` (N, 2, hips frame): how far the arms or waist moved the body's centre of mass; the hips shift
        the other way (up to ``max_com_shift_m``) to keep it over the feet."""
        plan = self.plan
        new = ~self._placed
        if new.any():  # standing: feet side by side under the hips
            self.feet_world_xy[new] = hips_xy[new].unsqueeze(1) + _rotate(self._nominal, hips_yaw[new, None])
            self.feet_world_yaw[new] = hips_yaw[new].unsqueeze(1).expand(-1, 2)
            self._was_stance[new] = True
            self._placed[new] = True

        period = cycle_period(plan, command)  # (N,)
        stance_time = period * plan.stance_fraction
        # how much the robot is walking (0 = standing): the caller ramps it up when walking starts and back down
        # after a stop (step_walk_in), so a stop finishes the step and brings the feet under the hips
        amount = walk_in.clamp(0.0, 1.0)
        foot_phase = torch.stack([phase, torch.remainder(phase + 0.5, 1.0)], dim=1)
        stance = (foot_phase < plan.stance_fraction) | (amount <= 0).unsqueeze(1)
        stance_progress = (foot_phase / plan.stance_fraction).clamp(0, 1)
        swing_progress = ((foot_phase - plan.stance_fraction) / (1 - plan.stance_fraction)).clamp(0, 1)
        smooth = (1 - torch.cos(torch.pi * swing_progress)) / 2

        lifting = self._was_stance & ~stance
        self._liftoff_xy = torch.where(lifting.unsqueeze(-1), self.feet_world_xy, self._liftoff_xy)
        self._liftoff_yaw = torch.where(lifting, self.feet_world_yaw, self._liftoff_yaw)

        # the middle of the planted feet, sideways from the hips (hips frame): the hips belong over it, and the
        # swinging foot lands half the foot spacing from it (forward and back, the hips stay where they are)
        planted = _rotate(self.feet_world_xy - hips_xy.unsqueeze(1), -hips_yaw.unsqueeze(1))
        planted = planted - _rotate(self._nominal.unsqueeze(0), self.feet_world_yaw - hips_yaw.unsqueeze(1))
        weights = stance.float()
        middle_y = (planted[..., 1] * weights).sum(1) / weights.sum(1).clamp(min=1.0)

        # where the hips will be when the swinging foot lands, and the landing spot half a step ahead of them
        remaining = (1 - swing_progress) * ((1 - plan.stance_fraction) * period * amount).unsqueeze(1)
        velocity = _rotate(command[:, :2], hips_yaw).unsqueeze(1)  # (N, 1, 2) world
        hips_at_landing = hips_xy.unsqueeze(1) + velocity * remaining.unsqueeze(-1)
        yaw_at_landing = hips_yaw.unsqueeze(1) + command[:, 2:3] * remaining
        half_turn = (0.5 * command[:, 2] * stance_time * amount).unsqueeze(1)
        half_step = (0.5 * command[:, :2] * (stance_time * amount).unsqueeze(1)).unsqueeze(1)  # (N, 1, 2) hips frame
        landing_relative = _rotate(self._nominal.unsqueeze(0), half_turn) + half_step
        landing_relative = landing_relative + torch.stack([torch.zeros_like(middle_y), middle_y], -1)[:, None]
        landing_xy = hips_at_landing + _rotate(landing_relative, yaw_at_landing)
        if hips_velocity is not None:
            # catch step (capture point): moving faster than commanded, in any direction, means tipping that way;
            # stepping into it stops the tip, the way a person catches a stumble
            surplus = hips_velocity - velocity[:, 0]
            catch = (plan.catch_gain_s * surplus).clamp(-plan.max_catch_step_m, plan.max_catch_step_m)
            landing_xy = landing_xy + (catch * amount.unsqueeze(1)).unsqueeze(1)
        landing_yaw = yaw_at_landing + half_turn

        turn_left = torch.remainder(landing_yaw - self._liftoff_yaw + torch.pi, 2 * torch.pi) - torch.pi
        swing_xy = self._liftoff_xy + (landing_xy - self._liftoff_xy) * smooth.unsqueeze(-1)
        swing_yaw = self._liftoff_yaw + turn_left * smooth
        self.feet_world_xy = torch.where(stance.unsqueeze(-1), self.feet_world_xy, swing_xy)
        self.feet_world_yaw = torch.where(stance, self.feet_world_yaw, swing_yaw)
        self._was_stance = stance

        relative_xy = _rotate(self.feet_world_xy - hips_xy.unsqueeze(1), -hips_yaw.unsqueeze(1))
        hips_offset = torch.stack([torch.zeros_like(middle_y), middle_y + hip_sway(plan, phase, amount)], dim=1)
        if com_shift is not None:
            hips_offset = hips_offset - com_shift.clamp(-plan.max_com_shift_m, plan.max_com_shift_m)
        relative_xy = relative_xy - hips_offset.unsqueeze(1)
        lift = torch.where(stance, torch.zeros_like(smooth), torch.sin(torch.pi * swing_progress))
        height = plan.stand_foot_z_m + (plan.foot_z_m - plan.stand_foot_z_m) * amount
        z = height.unsqueeze(1) + plan.step_height_m * lift * amount.unsqueeze(1)
        tilt = torch.where(
            stance, (1 - stance_progress / 0.15).clamp(0, 1), ((swing_progress - 0.6) / 0.4).clamp(0, 1)
        ) * plan.heel_tilt_rad * amount.unsqueeze(1)
        return FeetPlan(
            position=torch.cat([relative_xy, z.unsqueeze(-1)], dim=-1),
            rotation=_yaw_pitch_matrix(self.feet_world_yaw - hips_yaw.unsqueeze(1), tilt),
            stance=stance,
            hips_offset=hips_offset,
        )


def arm_motion(
    plan: GaitPlan, phase: torch.Tensor, command: torch.Tensor, walk_in: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shoulder pitch and elbow offsets (each (N, 2): left arm, right arm) that swing each arm with its own leg.

    Each arm follows where its leg is in the step (from the step clock, so the swing is clear at any speed): the
    right arm comes forward as the right leg does and goes back with it. The elbow bends as its arm comes forward
    and straightens as it goes back. On the R1, negative shoulder pitch is forward and a smaller elbow angle is
    more bent. Walking backward the arms follow the legs backward; turning swings them a little; standing, not at all.
    """
    foot_phase = torch.stack([phase, torch.remainder(phase + 0.5, 1.0)], dim=1)
    on_ground = foot_phase < plan.stance_fraction
    stance_progress = foot_phase / plan.stance_fraction
    swing_progress = ((foot_phase - plan.stance_fraction) / (1 - plan.stance_fraction)).clamp(0, 1)
    smooth = (1 - torch.cos(torch.pi * swing_progress)) / 2
    along = torch.where(on_ground, 0.5 - stance_progress, -0.5 + smooth)  # +0.5 = leg furthest in front
    forward_speed = command[:, 0]
    direction = torch.where(forward_speed < -plan.moving_threshold, -1.0, 1.0)  # walking backward: mirrored
    leg_forward = 2 * along * direction.unsqueeze(1)  # (N, 2): +1 = that leg furthest forward
    speed = torch.linalg.norm(command[:, :2], dim=1)
    walking_share = (speed / plan.full_swing_speed).clamp(0.0, 1.0)
    size = walk_in.clamp(0, 1) * (0.35 + 0.65 * walking_share.clamp(min=0.5) * (speed > plan.moving_threshold))
    shoulder = -plan.arm_swing_rad * size.unsqueeze(1) * leg_forward
    elbow = -plan.elbow_swing_rad * size.unsqueeze(1) * (leg_forward + 1) / 2
    return shoulder, elbow


def swing_on_arm_pose(
    commanded: torch.Tensor, rest: torch.Tensor, swing: torch.Tensor, gesture_rad: float = 0.2
) -> torch.Tensor:
    """Arm targets (N, 4: left and right shoulder pitch, then left and right elbow) for walking while gesturing.

    The walking swing is added to the commanded arm pose (from the gesture node). An arm that is away from its
    rest pose is gesturing, so its swing fades out (fully by ``gesture_rad``) and it follows the gesture exactly;
    the other arm keeps swinging.
    """
    away = torch.abs(commanded - rest)
    per_arm = torch.maximum(away[:, :2], away[:, 2:])  # (N, 2): how far each arm is from rest
    keep = (1 - per_arm / gesture_rad).clamp(0, 1).repeat(1, 2)
    return commanded + keep * swing


def move_toward(current: torch.Tensor, goal: torch.Tensor, max_step: float) -> torch.Tensor:
    """``current`` moved toward ``goal`` by at most ``max_step`` per element (a rate limit, so nothing jumps)."""
    return current + (goal - current).clamp(-max_step, max_step)


def without_gesture(
    joint_pos: torch.Tensor, joint_vel: torch.Tensor, gesture_offset: torch.Tensor, gesture_rate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint positions and velocities with the commanded gesture taken out (what the policy sees).

    A gesture moves the upper body's targets away from rest; taking that commanded part out leaves the swing and
    the tracking error, which the policy knows from walking, so a raised arm is not an input it has never seen.
    """
    return joint_pos - gesture_offset, joint_vel - gesture_rate


# --- the same planner in NumPy, for the CPU (one robot, live scene) --------------------------------------------------
# Torch spends ~5-10 us of bookkeeping on each tiny operation on the CPU, NumPy ~1 us; the planner is ~150 of them per
# step. The maths mirrors FootPlanter.step and arm_motion line for line (test_planner_equivalence.py checks them).

import numpy as np  # noqa: E402


def _np_rotate(xy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]], -1)


def _np_cycle_period(plan: GaitPlan, command: np.ndarray) -> np.ndarray:
    speed = np.linalg.norm(command[:, :2], axis=1)
    needed = plan.max_step_m / np.maximum(speed * plan.stance_fraction, 1e-6)
    return np.clip(needed, plan.min_period_s, plan.period_s)


def _np_yaw_pitch_matrix(yaw: np.ndarray, toe_up: np.ndarray) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(-toe_up), np.sin(-toe_up)
    out = np.zeros(yaw.shape + (3, 3))
    # Rz(yaw) @ Ry(-toe_up), multiplied out
    out[..., 0, 0], out[..., 0, 1], out[..., 0, 2] = cy * cp, -sy, cy * sp
    out[..., 1, 0], out[..., 1, 1], out[..., 1, 2] = sy * cp, cy, sy * sp
    out[..., 2, 0], out[..., 2, 1], out[..., 2, 2] = -sp, 0.0, cp
    return out


class NumpyFootPlanter:
    """FootPlanter for the CPU in NumPy: same state, same step(), torch tensors in and out."""

    def __init__(self, plan: GaitPlan, count: int) -> None:
        self.plan = plan
        self._nominal = np.stack([np.full(2, plan.foot_x_m), np.array([0.5, -0.5]) * plan.width_m], axis=1)
        self.feet_world_xy = np.zeros((count, 2, 2))
        self.feet_world_yaw = np.zeros((count, 2))
        self._liftoff_xy = np.zeros_like(self.feet_world_xy)
        self._liftoff_yaw = np.zeros_like(self.feet_world_yaw)
        self._was_stance = np.ones((count, 2), dtype=bool)
        self._placed = np.zeros(count, dtype=bool)

    def reset(self, ids=None) -> None:
        if ids is None:
            self._placed[:] = False
        else:
            self._placed[np.asarray(ids.cpu() if hasattr(ids, "cpu") else ids)] = False

    def step(self, hips_xy, hips_yaw, command, phase, walk_in, hips_velocity=None, com_shift=None) -> FeetPlan:
        def n(t):
            return None if t is None else t.detach().double().numpy()

        hips_xy, hips_yaw, command, phase, walk_in = n(hips_xy), n(hips_yaw), n(command), n(phase), n(walk_in)
        hips_velocity, com_shift = n(hips_velocity), n(com_shift)
        plan = self.plan
        new = ~self._placed
        if new.any():
            self.feet_world_xy[new] = hips_xy[new][:, None] + _np_rotate(self._nominal, hips_yaw[new, None])
            self.feet_world_yaw[new] = np.repeat(hips_yaw[new][:, None], 2, axis=1)
            self._was_stance[new] = True
            self._placed[new] = True

        period = _np_cycle_period(plan, command)
        stance_time = period * plan.stance_fraction
        amount = np.clip(walk_in, 0.0, 1.0)
        foot_phase = np.stack([phase, np.remainder(phase + 0.5, 1.0)], axis=1)
        stance = (foot_phase < plan.stance_fraction) | (amount <= 0)[:, None]
        stance_progress = np.clip(foot_phase / plan.stance_fraction, 0, 1)
        swing_progress = np.clip((foot_phase - plan.stance_fraction) / (1 - plan.stance_fraction), 0, 1)
        smooth = (1 - np.cos(np.pi * swing_progress)) / 2

        lifting = self._was_stance & ~stance
        self._liftoff_xy = np.where(lifting[..., None], self.feet_world_xy, self._liftoff_xy)
        self._liftoff_yaw = np.where(lifting, self.feet_world_yaw, self._liftoff_yaw)

        planted = _np_rotate(self.feet_world_xy - hips_xy[:, None], -hips_yaw[:, None])
        planted = planted - _np_rotate(self._nominal[None], self.feet_world_yaw - hips_yaw[:, None])
        weights = stance.astype(float)
        middle_y = (planted[..., 1] * weights).sum(1) / np.maximum(weights.sum(1), 1.0)

        remaining = (1 - swing_progress) * ((1 - plan.stance_fraction) * period * amount)[:, None]
        velocity = _np_rotate(command[:, :2], hips_yaw)[:, None]
        hips_at_landing = hips_xy[:, None] + velocity * remaining[..., None]
        yaw_at_landing = hips_yaw[:, None] + command[:, 2:3] * remaining
        half_turn = (0.5 * command[:, 2] * stance_time * amount)[:, None]
        half_step = (0.5 * command[:, :2] * (stance_time * amount)[:, None])[:, None]
        landing_relative = _np_rotate(self._nominal[None], half_turn) + half_step
        landing_relative = landing_relative + np.stack([np.zeros_like(middle_y), middle_y], -1)[:, None]
        landing_xy = hips_at_landing + _np_rotate(landing_relative, yaw_at_landing)
        if hips_velocity is not None:
            surplus = hips_velocity - velocity[:, 0]
            catch = np.clip(plan.catch_gain_s * surplus, -plan.max_catch_step_m, plan.max_catch_step_m)
            landing_xy = landing_xy + (catch * amount[:, None])[:, None]
        landing_yaw = yaw_at_landing + half_turn

        turn_left = np.remainder(landing_yaw - self._liftoff_yaw + np.pi, 2 * np.pi) - np.pi
        swing_xy = self._liftoff_xy + (landing_xy - self._liftoff_xy) * smooth[..., None]
        swing_yaw = self._liftoff_yaw + turn_left * smooth
        self.feet_world_xy = np.where(stance[..., None], self.feet_world_xy, swing_xy)
        self.feet_world_yaw = np.where(stance, self.feet_world_yaw, swing_yaw)
        self._was_stance = stance

        relative_xy = _np_rotate(self.feet_world_xy - hips_xy[:, None], -hips_yaw[:, None])
        sway = plan.hip_sway_m * amount * np.sin(2 * np.pi * (phase - 0.05))
        hips_offset = np.stack([np.zeros_like(middle_y), middle_y + sway], axis=1)
        if com_shift is not None:
            hips_offset = hips_offset - np.clip(com_shift, -plan.max_com_shift_m, plan.max_com_shift_m)
        relative_xy = relative_xy - hips_offset[:, None]
        lift = np.where(stance, 0.0, np.sin(np.pi * swing_progress))
        height = plan.stand_foot_z_m + (plan.foot_z_m - plan.stand_foot_z_m) * amount
        z = height[:, None] + plan.step_height_m * lift * amount[:, None]
        tilt = np.where(stance, np.clip(1 - stance_progress / 0.15, 0, 1),
                        np.clip((swing_progress - 0.6) / 0.4, 0, 1)) * plan.heel_tilt_rad * amount[:, None]

        def t(a, dtype=torch.float32):
            return torch.from_numpy(np.ascontiguousarray(a)).to(dtype)

        return FeetPlan(
            position=t(np.concatenate([relative_xy, z[..., None]], axis=-1)),
            rotation=t(_np_yaw_pitch_matrix(self.feet_world_yaw - hips_yaw[:, None], tilt)),
            stance=torch.from_numpy(stance.copy()),
            hips_offset=t(hips_offset),
        )


def np_arm_motion(plan: GaitPlan, phase, command, walk_in) -> tuple[torch.Tensor, torch.Tensor]:
    """arm_motion in NumPy (torch tensors in and out, on the CPU)."""
    phase, command, walk_in = (x.detach().double().numpy() for x in (phase, command, walk_in))
    foot_phase = np.stack([phase, np.remainder(phase + 0.5, 1.0)], axis=1)
    on_ground = foot_phase < plan.stance_fraction
    stance_progress = foot_phase / plan.stance_fraction
    swing_progress = np.clip((foot_phase - plan.stance_fraction) / (1 - plan.stance_fraction), 0, 1)
    smooth = (1 - np.cos(np.pi * swing_progress)) / 2
    along = np.where(on_ground, 0.5 - stance_progress, -0.5 + smooth)
    direction = np.where(command[:, 0] < -plan.moving_threshold, -1.0, 1.0)
    leg_forward = 2 * along * direction[:, None]
    speed = np.linalg.norm(command[:, :2], axis=1)
    walking_share = np.clip(speed / plan.full_swing_speed, 0.0, 1.0)
    size = np.clip(walk_in, 0, 1) * (0.35 + 0.65 * np.maximum(walking_share, 0.5) * (speed > plan.moving_threshold))
    shoulder = -plan.arm_swing_rad * size[:, None] * leg_forward
    elbow = -plan.elbow_swing_rad * size[:, None] * (leg_forward + 1) / 2
    return torch.from_numpy(shoulder).float(), torch.from_numpy(elbow).float()
