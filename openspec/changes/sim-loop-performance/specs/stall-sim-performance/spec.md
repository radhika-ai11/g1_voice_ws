# Spec Delta

## Purpose
Defines how fast the live GITEX stall simulation must run on the demo machine, how it reports its speed, and which behaviour must stay the same while it is made faster.

## ADDED Requirements

### Requirement: Real-time pacing
The stall simulation SHALL never run faster than real time, and SHALL aim to run at real time (100%) with the full demo running (simulation, voice assistant, live stall script) on compute-2. The minimum acceptance level for this change is 60% of real time with the full demo, sustained over 5 minutes.

#### Scenario: Full demo on compute-2
- **WHEN** the full stall demo runs on compute-2 for 5 minutes after startup
- **THEN** the median reported simulation speed is at least 60% of real time

#### Scenario: Faster than real time is capped
- **WHEN** a control step finishes in less than its real-time budget
- **THEN** the loop waits so simulated time does not run ahead of wall-clock time

### Requirement: Speed reporting
The simulation SHALL print its speed as a percentage of real time every 5 seconds, and when profiling is enabled it SHALL also print the average time per step for each major section (physics, foot planner and IK, viewer, visitors, camera image).

#### Scenario: Profiling enabled
- **WHEN** the simulation starts with `WALK_SCENE_PROFILE=1`
- **THEN** every 5 seconds it prints the speed and a per-section breakdown in milliseconds per step

### Requirement: Repeatable benchmark
The project SHALL provide a benchmark command that runs the stall scene for a fixed duration with a fixed scripted visitor, and prints the median speed and the per-section breakdown, so two versions of the code can be compared.

#### Scenario: Comparing two versions
- **WHEN** the benchmark runs twice on the same machine and code
- **THEN** the two median speeds differ by less than 10%

### Requirement: Walking behaviour unchanged
Performance changes SHALL NOT change how the robot walks. Leg joint angles from the foot planner and IK SHALL match the previous implementation within 1e-3 rad for the same inputs, both for one robot on the CPU and for a training batch on the GPU.

#### Scenario: IK equivalence
- **WHEN** the old and new IK solve the same recorded set of foot targets
- **THEN** every joint angle differs by at most 1e-3 rad

#### Scenario: No new falls
- **WHEN** the existing fall test (`training/test_falls.py`) runs on the new code
- **THEN** it reports no falls, as before

### Requirement: Scene and interface unchanged
The viewer SHALL show the same scene (hall, booths, walking visitors, stall visitor and held object), visitors SHALL stay solid to the robot, and the ROS 2 topics, message types and publish rates SHALL stay the same.

#### Scenario: Visitor stays solid
- **WHEN** the robot is commanded to walk into a visitor
- **THEN** it is blocked by the visitor's body, as before

#### Scenario: ROS interface
- **WHEN** the voice assistant runs against the new simulation
- **THEN** it receives the same topics (odometry, joint states, head camera image, nearby people, visit state) at the same rates
