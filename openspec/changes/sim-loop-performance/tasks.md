# Tasks

## 1. Measurement

- [x] 1.1 Add finer sections to the `WALK_SCENE_PROFILE=1` breakdown (planner, IK, PhysX, sensors, viewer, visitors, capsules, publishing, camera); verify the breakdown prints every 5 s and the sections add up to within 10% of the step time
- [x] 1.2 Add `training/bench_walk_scene.sh`: runs the GITEX scene headless and with viewer for 120 s with a scripted visit and walk, prints the median speed and the profile; verify two runs on compute-2 agree within 10%
- [x] 1.3 Record the baseline (full demo and sim alone) in `openspec/changes/sim-loop-performance/benchmarks.md`

## 2. Leg IK

- [x] 2.1 Add an IK equivalence test: record 2000 foot targets from a walk, solve with the old and new solver, assert max diff ≤ 1e-3 rad (CPU, N=1) and on a GPU batch of 4096
- [x] 2.2 (superseded by the NumPy solver) constants precomputed in NumPy; verified with 2.1 and the benchmark
- [x] 2.3 Add batch-wide early convergence exit to `solve`; verify with 2.1, the benchmark, and a training-throughput check (no slowdown, else limit it to the CPU plan device)

## 3. Visitors and viewer

- [x] 3.1 `PeopleBodies`: write only moved capsules, reuse a preallocated tensor; verify the capsule share in the profile drops and the robot is still blocked by a visitor
- [x] 3.2 `GitexHall`: re-log visitor meshes only when their frame or pose changed; cache normals per walk-cycle frame if `log_mesh` allows; verify the viewer looks the same and the viewer share drops
- [x] 3.3 Viewer redraw at `--viewer-hz` (wall-clock based, default 30); verify a smooth picture over Moonlight and the measured redraw rate

## 4. Using more cores

- [x] 4.1 Tune `torch.set_num_threads` with the benchmark (1, 2, 4, 16); keep the fastest
- [x] 4.2 Benchmark CPU PhysX (2/4/8 worker threads) vs GPU PhysX for the stall scene; make the faster the stall default; verify with `test_falls.py`
- [x] 4.3 Move visitor animation NumPy work and ROS publishing to a worker thread overlapping `env.step`; verify the speed gain in the benchmark and that ROS topic rates are unchanged (`ros2 topic hz`)

## 5. Decision point and phase 2

- [x] 5.1 Run the full-demo benchmark; if the median is ≥ 100%, skip 5.2–5.3
- [x] 5.2 (done differently) planner + IK in NumPy on the CPU instead of a fused GPU kernel: one robot is faster on the CPU; verified with test_ik_equivalence.py and test_planner_equivalence.py
- [x] 5.3 (only if still below target) Move the Newton viewer to its own process fed by shared memory; verify the same picture and the new benchmark numbers

## 6. Verification

- [x] 6.1 Full demo on compute-2 for 5 minutes: median speed ≥ 60% (target 100%); `test_falls.py` passes; visitor greeting and conversation work end to end
- [x] 6.2 Update `benchmarks.md` with before/after numbers and write a short summary for the developer
