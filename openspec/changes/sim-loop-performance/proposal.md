# Proposal

## Why

The GITEX stall demo (`training/walk_scene.py --gitex`) runs at about 19% of real time on compute-2 (16 vCPU Xeon Platinum 8362, L40-48Q), so the robot and visitors move in slow motion during partner demos. A py-spy profile of the full demo shows that the sim loop is bound by **one CPU thread**: the GPU is about half idle and 14 of 16 cores are unused. Most of the time goes to per-step Python/torch overhead around the physics, not to the physics itself:

| Part of one control step | Share of main-thread time |
|---|---|
| `env.step` in total | 53% |
| ├ foot planner + leg IK (`LegKinematics.solve`) | 15% |
| ├ PhysX step | 14.5% |
| └ visitor capsule pose writes (`PeopleBodies.update`) | 7.6% |
| Newton viewer (`update_visualizers`) | 21% |
| └ re-logging visitor meshes + recomputing normals | 12% |
| hall visitor animation (`GitexHall.step`) | ~8% |

## What Changes

- Add measurement tooling: a repeatable benchmark run (fixed duration, headless and with viewer) that reports real-time factor and a per-section breakdown, so every later change is judged by numbers.
- Cut per-step overhead on the main thread:
  - Leg IK: stop iterating once the foot is on target, and stop rebuilding constant tensors every call. Results must stay within a small tolerance of the current solver.
  - Visitor capsules: only write a capsule's pose when it moved, and stop allocating a tensor per capsule per step.
  - Newton viewer: re-upload visitor meshes only when their animation frame or position changed, and redraw at a configurable rate that is independent of the physics rate.
- Use more than one core: run the work that does not depend on the physics result (visitor animation, viewer mesh preparation, ROS publishing) in a worker thread that overlaps with the physics step. Measure CPU PhysX with several worker threads against GPU PhysX for this single-robot scene, and keep the faster one.
- If the in-process changes do not reach the target, move the Newton viewer into its own process fed by shared memory (phase 2, decided by measurement).
- No change to the trained policy, the training setup (thousands of robots on the GPU), the ROS 2 topics, or the voice assistant.

## Capabilities

### New Capabilities
- `stall-sim-performance`: how fast the live stall simulation runs, how it reports its speed, and the guarantee that speed work does not change how the robot walks, what the viewer shows, or the ROS 2 interface.

### Modified Capabilities
<!-- none: the project has no existing specs -->

## Impact

- Code: `training/walk_scene.py`, `training/foot_placement/leg_kinematics.py`, `training/walk_task/plan_action.py`, `training/people_sensor.py`, `training/gitex_hall.py`; a new benchmark script under `training/`.
- Training: `LegKinematics` is shared with training. Its batched GPU path must give the same results; the early exit applies only when the whole batch has converged.
- Runtime: uses more CPU cores on the machine that runs the sim (compute-2: up to 16).
- Owner: the project's developer (radhika) should review, since this changes their code.
