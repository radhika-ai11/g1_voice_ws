"""The NumPy foot planter and arm swing (CPU, live scene) give the same results as the torch ones (training).

Drives both with the same 2000-step walk: commands that change every 2 s (forward, backward, sideways, turning,
standing), hips that follow the command with a little noise, and the step clock and walk-in of the real action
term. Every foot position must agree within 1e-4 m, every foot rotation within 1e-4, stance exactly, and the arm
swing within 1e-5 rad.

    python training/test_planner_equivalence.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foot_placement.gait_planner import (  # noqa: E402
    FootPlanter, GaitPlan, NumpyFootPlanter, arm_motion, cycle_period, np_arm_motion, step_walk_in,
)


def main() -> None:
    plan, dt = GaitPlan(), 0.02
    generator = torch.Generator().manual_seed(1)
    torch_planter, numpy_planter = FootPlanter(plan, 1), NumpyFootPlanter(plan, 1)
    hips_xy, hips_yaw = torch.zeros(1, 2), torch.zeros(1)
    phase, walk_in = torch.full((1,), plan.stance_fraction), torch.zeros(1)
    worst = {"position": 0.0, "rotation": 0.0, "offset": 0.0, "arms": 0.0}
    stance_mismatch = 0
    command = torch.zeros(1, 3)
    for step in range(2000):
        if step % 100 == 0:  # a new command every 2 s; every 4th one is "stand"
            command = (torch.rand(1, 3, generator=generator) * 2 - 1) * torch.tensor([[0.8, 0.3, 0.8]])
            if (step // 100) % 4 == 3:
                command = torch.zeros(1, 3)
        moving = (torch.linalg.norm(command[:, :2], dim=1) > plan.moving_threshold) | (command[:, 2].abs() > plan.moving_threshold)
        walk_in = step_walk_in(walk_in, moving, dt, 0.8)
        ramp = 0.5 - 0.5 * torch.cos(torch.pi * walk_in)
        phase = torch.where(walk_in > 0, torch.remainder(phase + dt / cycle_period(plan, command), 1.0),
                            torch.full_like(phase, plan.stance_fraction))
        noise = 0.05 * torch.randn(1, 2, generator=generator)
        velocity = torch.stack([command[:, 0] * torch.cos(hips_yaw) - command[:, 1] * torch.sin(hips_yaw),
                                command[:, 0] * torch.sin(hips_yaw) + command[:, 1] * torch.cos(hips_yaw)], 1) + noise
        hips_xy, hips_yaw = hips_xy + velocity * dt, hips_yaw + command[:, 2] * dt
        com = 0.02 * torch.randn(1, 2, generator=generator)
        a = torch_planter.step(hips_xy, hips_yaw, command, phase, ramp, hips_velocity=velocity, com_shift=com)
        b = numpy_planter.step(hips_xy, hips_yaw, command, phase, ramp, hips_velocity=velocity, com_shift=com)
        worst["position"] = max(worst["position"], float((a.position - b.position).abs().max()))
        worst["rotation"] = max(worst["rotation"], float((a.rotation - b.rotation).abs().max()))
        worst["offset"] = max(worst["offset"], float((a.hips_offset - b.hips_offset).abs().max()))
        stance_mismatch += int((a.stance != b.stance).sum())
        (s1, e1), (s2, e2) = arm_motion(plan, phase, command, ramp), np_arm_motion(plan, phase, command, ramp)
        worst["arms"] = max(worst["arms"], float((s1 - s2).abs().max()), float((e1 - e2).abs().max()))
    limits = {"position": 1e-4, "rotation": 1e-4, "offset": 1e-4, "arms": 1e-5}
    failed = stance_mismatch > 0
    for name, value in worst.items():
        ok = value <= limits[name]
        failed |= not ok
        print(f"{name:9s} largest difference {value:.2e} {'OK' if ok else 'FAIL'} (limit {limits[name]:g})")
    print(f"stance    mismatches {stance_mismatch} {'OK' if stance_mismatch == 0 else 'FAIL'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
