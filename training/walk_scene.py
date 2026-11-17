"""Isaac Sim scene with a walking Unitree R1, driven by the trained policy and by ROS 2.

The robot is free (not pinned). The legs follow a footstep plan (IK), balanced by the trained policy, and each arm
swings with its own leg. /cmd_vel sets the walking speed and turning rate, and /robot/joint_command (from
gesture_node) moves the arms and waist, as in the standing scene; an arm doing a gesture stops swinging.

    ~/g1_voice_ws/start_isaac_walk.sh                    # latest trained policy
    ~/g1_voice_ws/start_isaac_walk.sh --checkpoint PATH  # a specific model_*.pt
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import os
import sys
import time

import warp as wp

wp.config.enable_backward = False

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAINING_DIR)
import walk_task  # noqa: E402,F401  (registers the task)
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args  # noqa: E402
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import check_rsl_rl_version  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config  # noqa: E402

PROFILE = os.environ.get("WALK_SCENE_PROFILE") == "1"
MAX_LAG_S = 0.25  # the pacing catches up at most this far behind real time
VIEWER_EVERY = 2  # with the head camera: redraw the Newton viewer every 2nd control step (25 Hz)
TASK = "R1-Walk-Plan-Flat-Play"  # foot planter + IK walk, balanced by the policy
# The tested policy (no falls in test_falls.py and test_gesture_balance.py); newer runs are not used until tested.
BEST_CHECKPOINT = "2026-11-05_10-59-32/model_5449.pt"


# Newton viewer cameras, as (eye, lookat): close to the robot, the whole GITEX hall from above its entrance corner,
# and the robot's Aideology stall from the aisle in front of it (the stall: x -2.8 to 2.2, y -4 to 4)
NEWTON_VIEWS = {
    "robot": ((3.0, -3.0, 1.6), (0.0, 0.0, 0.7)),
    "hall": ((30.0, -24.0, 22.0), (7.0, 0.0, 0.0)),
    "booth": ((5.2, -3.0, 2.4), (-0.4, 0.0, 1.0)),
    # from the side, inside the stall: the robot (at 0, 0) and the visitor talking to it (at 1.5, 0.27) side by side,
    # close up, where people walking along the aisle do not pass in front of the camera
    "stall": ((0.75, -2.6, 1.5), (0.75, 0.15, 0.85)),
}


def configure_for_deployment(env_cfg, viewer: str, follow_camera: bool, view: str = "robot") -> None:
    """One robot, commands from ROS instead of random ones, no time limit, and the chosen viewer."""

    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1.0e6
    env_cfg.events.animate_upper_body = None  # the arms and waist follow /robot/joint_command instead
    env_cfg.events.reset_base = None
    env_cfg.actions.joint_pos.plan_device = "cpu"  # one robot: the planner and IK are much faster on the CPU
    env_cfg.terminations.lost_balance = None  # training resets a robot early; here a stumble can be recovered
    env_cfg.terminations.hips_too_low = None
    # nor does touching something with the torso or head end an "episode" (that would teleport the robot back to its
    # start mid-demo; /scene/reset_robot does that on purpose); checking it also cost ~14% of every step
    env_cfg.terminations.base_contact = None
    for name in list(vars(env_cfg.rewards)):  # training rewards are not needed to walk, and cost time every step
        setattr(env_cfg.rewards, name, None)
    command = env_cfg.commands.base_velocity
    command.resampling_time_range = (1.0e9, 1.0e9)
    command.rel_standing_envs = 0.0
    command.debug_vis = False
    if viewer == "kit" and follow_camera:
        from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg

        env_cfg.sim.visualizer_cfgs = [
            KitVisualizerCfg(origin_type="asset", origin_track_path="robot", eye=(2.4, -1.2, 0.9), lookat=(0.0, 0.0, 0.1))
        ]
    elif viewer == "newton_gl":
        from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonGLVisualizerCfg

        eye, lookat = NEWTON_VIEWS[view]
        env_cfg.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(window_width=1600, window_height=900, eye=eye, lookat=lookat)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="", help="model_*.pt to load (default: BEST_CHECKPOINT)")
    parser.add_argument("--headless", action="store_true", help="run without any window (same as --viewer none)")
    parser.add_argument(
        "--viewer",
        choices=["newton_gl", "kit", "none"],
        default="newton_gl",
        help="newton_gl: light OpenGL window (fast); kit: full Isaac Sim window (slow on some laptops)",
    )
    parser.add_argument("--surroundings", action="store_true",
                        help="head camera on ROS, people walking past and objects around (surroundings.py)")
    parser.add_argument("--gitex", action="store_true",
                        help="the GITEX exhibition hall (gitex_hall.py): booths, products, visitors walking, the head "
                             "camera on ROS; everything also drawn in the Newton viewer, seen whole from above")
    parser.add_argument("--view", choices=["hall", "booth", "stall"], default="hall",
                        help="with --gitex: the Newton viewer camera: the whole hall, the robot's stall, or close "
                             "on the robot and its visitor")
    parser.add_argument("--visitors", type=int, default=10, help="with --gitex: how many visitors walk the hall")
    parser.add_argument("--people", type=int, default=2, help="with --surroundings: how many people walk past")
    parser.add_argument("--no-follow", action="store_true", help="do not move the camera with the robot")
    parser.add_argument("--viewer-in-process", action="store_true",
                        help="with --gitex: draw the Newton viewer in the simulation's own process (slower; by default "
                             "it runs in a process of its own, viewer_process.py)")
    parser.add_argument("--viewer-fps", type=float, default=30.0,
                        help="with --gitex: how often the robot and visitors are sent to the viewer (per real second)")
    own_args, launcher_args = parser.parse_known_args()

    viewer = "none" if own_args.headless else own_args.viewer
    # the GITEX hall's Newton window runs in its own process (viewer_process.py), on its own CPU core
    remote_viewer = own_args.gitex and viewer == "newton_gl" and not own_args.viewer_in_process
    if remote_viewer:
        viewer = "none"
    argv = ["--task", TASK, "--num_envs", "1", "--viz", viewer, *launcher_args]
    if not any(arg.startswith("--device") for arg in launcher_args):
        # one robot: PhysX on the CPU takes ~7 ms per step, on the GPU ~36 ms (tiny GPU jobs cost more to launch
        # than they compute); pass --device cuda:0 for the GPU
        argv += ["--device", "cpu"]
    if not any("numThreads" in arg for arg in launcher_args):
        # 4 PhysX worker threads: as fast as more for one robot, and they leave the other cores to the main loop
        # (with the default count the workers crowded the main thread out for ~30 s at a time: 75% -> 20%)
        argv += ["--kit_args=--/physics/numThreads=4"]
    args_cli = _parse_args([*argv, "physics=isaacsim_physx"])
    if own_args.surroundings or own_args.gitex:
        args_cli.enable_cameras = True  # the head camera needs rendering, also without a window
    check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
    configure_for_deployment(env_cfg, viewer, follow_camera=not own_args.no_follow, view=own_args.view if own_args.gitex else "robot")
    if own_args.surroundings or own_args.gitex:
        from surroundings import CAMERA_PERIOD_S, configure_head_camera

        configure_head_camera(env_cfg)
    if own_args.gitex:
        from people_sensor import configure_people_bodies

        configure_people_bodies(env_cfg, own_args.visitors + 1)  # every visitor is solid, the stall visitor too
        # Isaac Sim's renderer only renders as often as the head camera needs (5 Hz, not at every 50 Hz control
        # step): rendering every step kept the GPU so busy it overheated and slowed to a crawl. The Newton viewer
        # is still updated every control step (below).
        env_cfg.sim.render_interval = round(CAMERA_PERIOD_S / env_cfg.sim.dt)

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        import torch

        if args_cli.device == "cpu":
            torch.set_num_threads(1)  # one robot: tiny tensors, where handing work to other threads costs more

        from heading_hold import HeadingHold
        from walk_ros_bridge import WalkRosBridge
        from walk_task.reward_math import heading_from_quat_xyzw
        from walk_task.plan_action import ARM_SWING_JOINTS

        agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli), check_rsl_rl_version())
        apply_env_overrides(args_cli, env_cfg)
        log_root = os.path.join(TRAINING_DIR, "logs", "rsl_rl", agent_cfg.experiment_name)
        best = os.path.join(log_root, BEST_CHECKPOINT)
        checkpoint = own_args.checkpoint or (best if os.path.exists(best) else get_checkpoint_path(log_root, ".*", r"model_.*\.pt"))
        print(f"[walk_scene] Loading policy {checkpoint}", flush=True)

        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        command_term = base_env.command_manager.get_term("base_velocity")
        walk_term = base_env.action_manager.get_term("joint_pos")  # swings the arms; gestures ride on it
        joint_names = list(robot.joint_names)
        joint_index = {name: i for i, name in enumerate(joint_names)}
        env_ids = torch.tensor([0], device=base_env.device)
        bridge = WalkRosBridge()
        bridge.spin_in_background()
        heading_hold = HeadingHold()  # walking straight keeps the heading it started with
        cleanup.callback(bridge.shutdown)
        surroundings = camera = hall = None
        if own_args.gitex:
            from gitex_hall import GitexHall
            from surroundings import Surroundings

            origin = base_env.scene.env_origins[0, :2].tolist()
            newton = next((v for v in base_env.sim.visualizers if hasattr(v, "log_mesh")), None)
            if remote_viewer:
                from viewer_process import RemoteViewer

                newton = RemoteViewer(list(robot.body_names), *NEWTON_VIEWS[own_args.view])
                cleanup.callback(newton.close)
            hall = GitexHall(base_env.sim.stage, (origin[0], origin[1]), newton=newton, visitors=own_args.visitors)
            surroundings = Surroundings(base_env.sim.stage, (origin[0], origin[1]), people=0, room=False)
            camera = base_env.scene.sensors["head_camera"]
            last_image_time = -1.0
            from people_sensor import PeopleBodies, robot_frame

            bodies = PeopleBodies(base_env.scene, own_args.visitors + 1)
            people_time = -1.0
            print(f"[walk_scene] GITEX hall ready ({own_args.visitors} visitors; Newton viewer: "
                  f"{'yes' if newton is not None else 'no'}).", flush=True)
        elif own_args.surroundings:
            from surroundings import Surroundings

            origin = base_env.scene.env_origins[0, :2].tolist()
            surroundings = Surroundings(base_env.sim.stage, (origin[0], origin[1]), people=own_args.people)
            camera = base_env.scene.sensors["head_camera"]
            last_image_time = -1.0

        for visualizer in base_env.sim.visualizers:  # the window must not wait for the screen's refresh (vsync):
            window = getattr(visualizer, "_viewer", None)  # the whole sim loop would wait with it, up to 16.7 ms
            if window is not None and hasattr(window, "vsync"):
                window.vsync = False
        step_dt = base_env.step_dt
        sim_time = 0.0
        visit_state, visit_state_time = "", 0.0
        obs = env.get_observations()
        report_wall, report_sim = time.monotonic(), 0.0
        timings: dict[str, float] = collections.defaultdict(float)
        if PROFILE:  # the foot planner + IK run inside env.step: time them on their own
            def timed(name, function):
                def wrapper(*args, **kwargs):
                    mark = time.monotonic()
                    try:
                        return function(*args, **kwargs)
                    finally:
                        timings[name] += time.monotonic() - mark
                return wrapper

            walk_term.process_actions = timed("(in env step) planner and IK", walk_term.process_actions)
            base_env.sim.step = timed("(in env step) physics", base_env.sim.step)
            base_env.sim.render = timed("(in env step) render", base_env.sim.render)
        step_count = 0
        wall_start = None  # wall-clock time at sim time 0, for the real-time pacing
        last_viewer_send = 0.0
        steps_per_render = max(1, env_cfg.sim.render_interval // env_cfg.decimation)
        print("[walk_scene] Walking. Listening on /cmd_vel and /robot/joint_command.", flush=True)
        def reset_robot() -> None:
            """Start the robot again from its place at the stall, standing (a new round of the stall demo)."""
            with torch.inference_mode():  # the walking policy's tensors were made in inference mode
                base_env.reset(env_ids=env_ids)  # the walking policy's own state, too
                pose = robot.data.default_root_pose.torch[0:1].clone()
                pose[:, :3] += base_env.scene.env_origins[0:1]
                robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
                robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base_env.device),
                                                       env_ids=env_ids)
                joints = robot.data.default_joint_pos.torch[0:1].clone()
                robot.write_joint_position_to_sim_index(position=joints, env_ids=env_ids)
                robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joints), env_ids=env_ids)
            print("[walk_scene] Robot back at its place at the stall.", flush=True)

        while True:
            started = time.monotonic()
            bridge.spin_once()
            if bridge.reset_requested():
                reset_robot()
                heading_hold = HeadingHold()
                obs = env.get_observations()
            forward, left, turn = bridge.velocity_command()
            yaw = float(heading_from_quat_xyzw(robot.data.root_quat_w.torch[0:1])[0])
            turn = heading_hold.turn_command((forward, left, turn), yaw)
            command_term.vel_command_b[0] = torch.tensor([forward, left, turn], device=base_env.device)
            targets = {name: value for name, value in bridge.upper_body_targets().items() if name in joint_index}
            if all(name in targets for name in ARM_SWING_JOINTS):
                walk_term.set_arm_pose(torch.tensor([[targets.pop(name) for name in ARM_SWING_JOINTS]],
                                                    device=base_env.device))
            if targets:
                walk_term.set_gesture_targets([joint_index[name] for name in targets],
                                              torch.tensor([list(targets.values())], device=base_env.device))
                robot.actuators.target_command.set_position_index(
                    value=torch.tensor([list(targets.values())], device=base_env.device),
                    joint_ids=[joint_index[name] for name in targets],
                    env_ids=env_ids,
                )
            mark = time.monotonic()
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
            sim_time += step_dt
            timings["env step"] += time.monotonic() - mark
            step_count += 1
            render_next = step_count % steps_per_render == steps_per_render - 1  # the camera renders next step
            if remote_viewer:  # send the robot and visitors at the viewer's rate, in real (wall-clock) time
                viewer_now = newton.alive() and time.monotonic() - last_viewer_send >= 1.0 / own_args.viewer_fps
                if viewer_now:
                    mark = time.monotonic()
                    newton.publish_robot(robot.data.body_pose_w.torch[0].cpu().numpy())
                    last_viewer_send = mark
                    timings["viewer (send)"] += time.monotonic() - mark
            else:
                viewer_now = step_count % VIEWER_EVERY == 0
            if camera is not None and viewer_now and not remote_viewer:  # rendering is sparse (render_interval): update the viewer here
                mark = time.monotonic()
                base_env.sim.update_visualizers(step_dt * VIEWER_EVERY, skip_app_pumping=True)
                timings["newton viewer"] += time.monotonic() - mark

            mark = time.monotonic()
            data = robot.data
            bridge.publish(
                sim_time,
                joint_names,
                data.joint_pos.torch[0].tolist(),
                data.root_pos_w.torch[0].tolist(),
                data.root_quat_w.torch[0].tolist(),
                data.root_lin_vel_b.torch[0].tolist(),
                data.root_ang_vel_b.torch[0].tolist(),
            )
            timings["ros publish"] += time.monotonic() - mark
            if surroundings is not None:
                surroundings.step(step_dt)
                requests = bridge.show_requests()
                for name in requests:
                    robot_xy = tuple(robot.data.root_pos_w.torch[0, :2].tolist())
                    print(f"[walk_scene] Showing: {surroundings.show(name, robot_xy, yaw)}", flush=True)
                if hall is not None:  # after the show requests, so the viewer gets what is on the stand now
                    for request in bridge.visit_requests():
                        if request == "leave":
                            hall.end_visit()
                        elif not hall.request_visit(request):
                            print(f"[walk_scene] Visit not started ({request!r}): {hall.visit_state()}", flush=True)
                    mark = time.monotonic()
                    hall.robot_xy = tuple(robot.data.root_pos_w.torch[0, :2].tolist())
                    hall.step(step_dt, shown_changed=bool(requests), stage=render_next, viewer=viewer_now)
                    people = hall.people_xy()
                    timings["hall (visitors)"] += time.monotonic() - mark
                    mark = time.monotonic()
                    bodies.update(people, tuple(robot.data.root_pos_w.torch[0, :2].tolist()))  # solid, where drawn
                    timings["visitor bodies"] += time.monotonic() - mark
                    mark = time.monotonic()
                    if sim_time - people_time >= 0.1:  # the range sensor, 10 times a second
                        robot_xy = tuple(robot.data.root_pos_w.torch[0, :2].tolist())
                        bridge.publish_people(robot_frame(people, robot_xy, yaw))
                        people_time = sim_time
                    timings["hall (visitors)"] += time.monotonic() - mark
                    if (state := hall.visit_state()) != visit_state or sim_time - visit_state_time >= 1.0:
                        if state != visit_state:
                            print(f"[walk_scene] Stall visitor: {state}", flush=True)
                        bridge.publish_visit_state(state)  # also once a second, for listeners that join late
                        visit_state, visit_state_time = state, sim_time
                if sim_time - last_image_time >= camera.cfg.update_period:
                    mark = time.monotonic()
                    bridge.publish_image_in_background(sim_time, camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy())
                    last_image_time = sim_time
                    timings["camera image"] += time.monotonic() - mark
            if time.monotonic() - report_wall >= 5.0:
                speed = (sim_time - report_sim) / (time.monotonic() - report_wall)
                print(f"[walk_scene] simulation speed {speed:.0%} of real time", flush=True)
                if PROFILE:  # where the time of a step goes (WALK_SCENE_PROFILE=1)
                    steps = max(1, round((sim_time - report_sim) / step_dt))
                    print("[walk_scene] per step: " + ", ".join(f"{k} {1000 * v / steps:.1f} ms" for k, v in timings.items()),
                          flush=True)
                timings.clear()
                report_wall, report_sim = time.monotonic(), sim_time
            timings["whole step"] += time.monotonic() - started
            # Keep to real time on average: sleep only while ahead of the wall clock. A slow step (the head camera
            # renders every 10th step) is made up by the next ones; after a long stall (more than MAX_LAG_S behind)
            # the clock is moved on instead, so the scene never fast-forwards. Never faster than real time.
            now = time.monotonic()
            if wall_start is None or now - (wall_start + sim_time) > MAX_LAG_S:
                wall_start = now - sim_time
            ahead = wall_start + sim_time - now
            if ahead > 0:
                time.sleep(ahead)


if __name__ == "__main__":
    main()
