# Design

## Context

`walk_scene.py` runs one loop on the main Python thread. Each 20 ms control step does, in order: ROS spin, command handling, `env.step` (planner + IK on the CPU, then PhysX substeps, sensors, observations, policy), viewer update every 2nd step, ROS publishing, visitor animation, capsule writes, camera image every 0.2 s. Everything runs in sequence under the Python GIL, so only one core is busy. A step currently takes about 105 ms against a 20 ms budget. The baseline numbers are in proposal.md.

Constraints:
- The same `LegKinematics` and `PlanFollowingAction` are used in training with thousands of robots on the GPU. That path must not get slower or different.
- Isaac Lab 3.0 / Isaac Sim 6.2 / Newton viewer are external code and are not modified. Changes stay in this repo.
- The demo runs in Docker container `r1` on compute-2, which has 16 vCPUs.

## Goals / Non-Goals

**Goals:**
- Reach real time with the full demo if possible, and at least 60% (spec).
- Every change is measured with the benchmark before and after, and kept only if it helps.
- Use extra cores where work can really run in parallel.

**Non-Goals:**
- Retraining the policy or changing the gait.
- Changing the voice assistant's models or where they run (separate topic).
- Modifying Isaac Lab, Isaac Sim or Newton source code.

## Decisions

**D1. Measure first, with a fixed benchmark.** A `training/bench_walk_scene.sh` runs the GITEX scene for 120 s with a scripted visit and a scripted walk command, then prints the median speed and the profile. The profile gets finer sections (planner, IK, PhysX, sensors, viewer, visitors, capsules, publishing). *Alternative:* ad-hoc py-spy runs. These are kept for deep dives, but they aren't repeatable enough to compare small changes.

**D2. IK: converge early and keep constants.** `solve` stops once the largest position and rotation error in the whole batch is below a tolerance (about 1e-4). The warm start means it usually needs 1–2 iterations instead of 4. The per-joint constant tensors (`axis`, origins, identity, the skew matrix of each axis) are built once in `__init__`, and `axis_angle_matrix` uses the Rodrigues form with precomputed `axis·axisᵀ` and skew. For training, the batch-wide check is one GPU→CPU sync per iteration; if the training benchmark shows it slows down, the early exit is enabled only for the CPU plan device. *Alternative:* a hand-written NumPy solver for N=1. That is faster per op but duplicates the maths, and the two versions could drift apart.

**D3. Capsules: write only what changed.** `PeopleBodies` keeps each capsule's last position, writes only capsules that moved more than 1 mm, and reuses one preallocated pose tensor. Parked visitors cost nothing. If several move, one batched write is used when the asset API supports it. *Alternative:* fewer physics updates for the capsules. Rejected, because visitors would then be solid in the wrong place.

**D4. Viewer: upload less, at its own rate.** Visitor meshes are re-logged only when the animation frame or the pose changed since the last upload. Per-frame vertex normals are cached for the walk-cycle frames, so the viewer doesn't recompute them, if the Newton `log_mesh` API accepts normals; otherwise, the uploads are just made less often. The viewer redraw rate becomes a setting (`--viewer-hz`, default 30) that depends on wall-clock time, not on step count. *Alternative:* a lower fixed redraw rate. This is part of the solution, but alone it makes the picture look choppy.

**D5. Overlap work with the physics step.** PhysX, Warp launches and the OpenGL driver release the GIL while they work. A single worker thread prepares the next visitor animation frame (NumPy, which also releases the GIL) and does the ROS publishing while the main thread runs `env.step`. Everything that touches the USD stage or the Newton viewer stays on the main thread, because neither is thread-safe. `torch.set_num_threads` is tuned with the benchmark: tiny tensors usually run faster on 1–2 threads than on 16. *Alternative:* `multiprocessing` for everything. Rejected for the first phase because the robot state would have to be copied between processes every step.

**D6. Pick the PhysX device by measurement.** For one robot and 11 kinematic capsules, CPU PhysX with 4–8 worker threads may beat the GPU, because a GPU step for a scene this small costs mostly launch and sync overhead. The benchmark runs both, and the faster one becomes the stall default (the training default stays on the GPU).

**D6b. Phase 2, only if below target: fused GPU planner/IK.** The planner and IK maths are rewritten as a single Warp kernel (or the existing torch code is captured once as a CUDA Graph and replayed), so a step costs one GPU launch instead of ~2,000. This also speeds up training. It comes after the quick wins because it is more work and touches shared training code.

**D7. Phase 2, only if still below target: viewer in its own process.** The Newton viewer process reads the robot pose, joint angles and visitor poses from shared memory (a small fixed-size buffer written once per step) and draws at its own rate on its own core. This removes the viewer's 21% from the sim loop completely. It is decided after phase 1 by the benchmark.

## Risks / Trade-offs

- [Early IK exit changes gait slightly] → equivalence test against the old solver (1e-3 rad), plus `test_falls.py`.
- [Training gets slower from the convergence check] → benchmark training throughput; limit the early exit to the CPU path if needed.
- [Threads cause race conditions with USD/Newton] → only pure NumPy and ROS publishing go to the worker. USD and viewer calls stay on the main thread.
- [CPU PhysX behaves slightly differently from GPU PhysX] → `test_falls.py` and a visual check on both. Fall back to the GPU if the robot is less stable.
- [Real time is not reached even after phase 2] → report the measured ceiling (the PhysX step alone is ~15 ms per step) and the remaining options (fewer substeps, simpler collision shapes) to the developer, rather than changing the physics silently.

## Migration Plan

All work is on branch `perf/sim-loop`. Each step is one commit, with its benchmark numbers in the commit message. Testing runs in container `r1` on compute-2. To roll back, check out `main`; new settings (`--viewer-hz`, PhysX device) default to safe values.

## Open Questions

- Does Newton's `log_mesh` accept precomputed normals? This only changes how D4 saves time, not the plan.
