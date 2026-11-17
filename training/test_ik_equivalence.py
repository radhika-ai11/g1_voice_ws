"""The leg IK gives the same joint angles as before the speed work (training/foot_placement/leg_kinematics.py).

Solves a walking sequence of foot targets twice, warm-started step to step as in the walk: with the solver as it is
now, and with the original solver (from git, branch main). Every joint angle must agree within 1e-3 rad.

    python training/test_ik_equivalence.py [--ref main]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOLERANCE_RAD = 1e-3


def load_reference(ref: str):
    """The leg_kinematics module as it is on git ``ref``."""
    source = subprocess.run(["git", "-C", HERE, "show", f"{ref}:training/foot_placement/leg_kinematics.py"],
                            check=True, capture_output=True, text=True).stdout
    path = os.path.join(tempfile.mkdtemp(), "leg_kinematics_ref.py")
    with open(path, "w") as handle:
        handle.write(source)
    spec = importlib.util.spec_from_file_location("leg_kinematics_ref", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def walking_targets(leg, steps: int, batch: int, device: str):
    """Foot targets along a smooth stepping motion around the standing pose (step-to-step changes like a walk)."""
    start_position, start_rotation = leg.foot_pose(leg.default_angles().to(device))
    phases = torch.linspace(0, 1, batch, device=device)[:, None]
    for step in range(steps):
        t = 2 * math.pi * (step * 0.02 / 0.8 + phases)  # a 0.8 s gait cycle at 50 Hz
        offset = torch.cat([0.12 * torch.sin(t), 0.02 * torch.sin(t), 0.05 * torch.clamp(torch.sin(t), min=0)], 1)
        yield start_position + offset, start_rotation.expand(batch, 3, 3)


def run(module, device: str, batch: int, steps: int, iterations: int) -> torch.Tensor:
    urdf = os.path.expanduser("~/g1_voice_ws/robots/r1/R1.urdf")
    leg = module.LegKinematics.from_urdf(urdf, "left").to(device)
    angles = leg.default_angles().expand(batch, 6).clone()
    results = []
    for position, rotation in walking_targets(leg, steps, batch, device):
        angles = leg.solve(position, rotation, angles, iterations=iterations, safe=True)
        results.append(angles.clone())
    return torch.stack(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="main", help="git ref with the original solver")
    args = parser.parse_args()
    import foot_placement.leg_kinematics as current

    reference = load_reference(args.ref)
    cases = [("cpu", 1, 2000)]
    if torch.cuda.is_available():
        cases.append(("cuda", 4096, 200))
    failed = False
    for device, batch, steps in cases:
        new = run(current, device, batch, steps, iterations=4)
        old = run(reference, device, batch, steps, iterations=4)
        worst = float((new - old).abs().max())
        ok = worst <= TOLERANCE_RAD
        failed |= not ok
        print(f"{device:4s} batch {batch:5d}, {steps} steps: largest difference {worst:.2e} rad "
              f"{'OK' if ok else 'FAIL'} (limit {TOLERANCE_RAD:g})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
