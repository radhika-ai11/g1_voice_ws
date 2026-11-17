# Benchmarks: GITEX stall scene on compute-2

Measured with `training/bench_walk_scene.sh 120` (scene alone, scripted walk + one stall visit, median of 5 s
reports) unless noted. compute-2: 16 vCPU Xeon Platinum 8362, L40-48Q, inside container `r1`.

| Step | Median speed | Step time |
|---|---|---|
| Baseline (`main`) | 12% (5-21%) | 97-185 ms |
| Viewer index buffers once, capsules only when moved, IK early exit | 20% | 102 ms |
| PhysX on the CPU (`--device cpu`), host RAM freed | 33% | 60 ms |
| Visitor normals precomputed | 37% | 53 ms |
| Newton viewer in its own process (`viewer_process.py`) | 41% | 49 ms |
| NumPy IK, camera image published from a worker thread | 50% | 40 ms |
| Visitor meshes built in the viewer process | ~63% | 32 ms |
| `base_contact` termination off, far capsules 5 Hz | 67.5% | 28 ms |
| Host: `vm.compaction_proactiveness=0` (kcompactd TLB-shootdown storms) | 78% (73-80%) | 22.8 ms |
| Catch-up pacing, torch 1 thread | 90% | 22.0 ms |
| ROS callbacks on a thread, far visitor poses every 4th camera frame | 93% | 21.3 ms |
| NumPy foot planter and arm swing | 95% | 21.0 ms |
| Head camera 3.3 Hz | **100% (98-101%)** | 19.3 ms |
| **Full demo** (assistant, vision, Whisper, live stall), 5 min | **100% (93-104%)** | |

The swings between ~20% and ~80% seen until the host fix were not the sim: the host kernel's proactive memory
compaction (`kcompactd0`) migrated compute-2's pages, sending ~1.25 M TLB-shootdown interrupts per second to its
vCPUs (found with ftrace on `flush_tlb_mm_range`).

Checks: `test_ik_equivalence.py` (CPU 1.4e-4 rad, GPU bit-identical), `test_planner_equivalence.py` (4e-7),
`test_falls.py --deployment` on CPU physics: 0/240 falls (same tilt-limit failure as on the GPU, from the policy).
