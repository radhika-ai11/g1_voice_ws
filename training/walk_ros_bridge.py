"""ROS 2 side of the walking scene: velocity commands and arm/waist targets in, clock, odometry and joints out."""

from __future__ import annotations

import json

import threading
import time
from collections.abc import Sequence

import rclpy
import rclpy.executors
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String

# Stop walking if no velocity command arrives for this long (for example if the motion node crashed).
COMMAND_TIMEOUT_S = 0.5
MAX_COMMAND = (1.0, 0.3, 1.0)  # forward m/s, sideways m/s, turning rad/s: the ranges the policy was trained on
MIN_COMMAND = (-0.5, -0.3, -1.0)


class WalkRosBridge:
    def __init__(self) -> None:
        """The simulation loop must call spin_once() regularly to handle incoming messages."""
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("robot_walk_scene")
        self._lock = threading.Lock()
        self._velocity = (0.0, 0.0, 0.0)
        self._velocity_time = 0.0
        self._upper_body: dict[str, float] = {}
        self._show_requests: list[str] = []
        self._visit_requests: list[str] = []

        self._node.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self._node.create_subscription(JointState, "/robot/joint_command", self._on_joint_command, 10)
        self._clock_pub = self._node.create_publisher(Clock, "/clock", 10)
        self._joint_pub = self._node.create_publisher(JointState, "/robot/joint_states", 10)
        self._odom_pub = self._node.create_publisher(Odometry, "/robot/odom", 10)
        self._image_pub = self._node.create_publisher(Image, "/robot/camera/image_raw", 2)
        # "apple", "banana", ... or "none": hold that object up in front of the robot's camera (surroundings.py)
        self._node.create_subscription(String, "/scene/show", self._on_show, 10)
        # GITEX hall: "apple", "banana", ... sends a visitor to the robot's stall holding it; "leave" sends them away
        self._node.create_subscription(String, "/gitex/visit", self._on_visit, 10)
        self._visit_state_pub = self._node.create_publisher(String, "/gitex/visit_state", 10)
        # the people around the robot in its own frame, as a range sensor reports them (JSON list of [x, y])
        self._people_pub = self._node.create_publisher(String, "/robot/nearby_people", 10)
        self._resets = 0
        self._node.create_subscription(String, "/scene/reset_robot", self._on_reset, 10)
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self._node)

    def spin_once(self) -> None:
        """Handle pending ROS messages without waiting (nothing to do once spin_in_background() was called)."""
        if getattr(self, "_spin_thread", None) is None:
            self._executor.spin_once(timeout_sec=0.0)

    def spin_in_background(self) -> None:
        """Handle incoming messages on a thread of their own: polling for them cost the sim loop ~1 ms per step.
        The callbacks only store what arrived, under the lock; the loop reads it with the getters as before."""
        self._spin_thread = threading.Thread(target=self._spin, name="ros-spin", daemon=True)
        self._spin_thread.start()

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception:  # shutting down
            pass

    def _on_cmd_vel(self, msg: Twist) -> None:
        command = (msg.linear.x, msg.linear.y, msg.angular.z)
        clamped = tuple(min(max(value, low), high) for value, low, high in zip(command, MIN_COMMAND, MAX_COMMAND))
        with self._lock:
            self._velocity = clamped
            self._velocity_time = time.monotonic()

    def _on_show(self, msg: String) -> None:
        with self._lock:
            self._show_requests.append(msg.data)

    def _on_visit(self, msg: String) -> None:
        with self._lock:
            self._visit_requests.append(msg.data.strip().lower())

    def _on_reset(self, _msg: String) -> None:
        with self._lock:
            self._resets += 1

    def reset_requested(self) -> bool:
        """Whether a new round asked for the robot to start again from its place at the stall."""
        with self._lock:
            requested, self._resets = self._resets > 0, 0
        return requested

    def publish_people(self, people: list[tuple[float, float]]) -> None:
        self._people_pub.publish(String(data=json.dumps([[round(x, 3), round(y, 3)] for x, y in people])))

    def visit_requests(self) -> list[str]:
        """Stall visits asked for on /gitex/visit since the last call."""
        with self._lock:
            requests, self._visit_requests = self._visit_requests, []
        return requests

    def publish_visit_state(self, state: str) -> None:
        """Where the stall visitor is: "idle", "approaching apple", "at_stall apple", "leaving apple"."""
        self._visit_state_pub.publish(String(data=state))

    def show_requests(self) -> list[str]:
        """Objects asked for on /scene/show since the last call."""
        with self._lock:
            requests, self._show_requests = self._show_requests, []
        return requests

    def publish_image_in_background(self, sim_time_s: float, rgb) -> bool:
        """publish_image on a worker thread, so the simulation does not wait for the ~1 MB copy and serialisation.
        Returns False (the image is dropped) while the previous one is still being sent."""
        import concurrent.futures

        if getattr(self, "_image_worker", None) is None:
            self._image_worker = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="camera-publish")
            self._image_job = None
        if self._image_job is not None and not self._image_job.done():
            return False
        self._image_job = self._image_worker.submit(self.publish_image, sim_time_s, rgb)
        return True

    def publish_image(self, sim_time_s: float, rgb) -> None:
        """Publish a head camera image (H x W x 3 uint8 array) as sensor_msgs/Image (rgb8)."""
        msg = Image()
        msg.header.stamp.sec, msg.header.stamp.nanosec = int(sim_time_s), int((sim_time_s % 1.0) * 1e9)
        msg.header.frame_id = "head_camera"
        msg.height, msg.width = int(rgb.shape[0]), int(rgb.shape[1])
        msg.encoding, msg.step = "rgb8", int(rgb.shape[1]) * 3
        msg.data = rgb.tobytes()
        self._image_pub.publish(msg)

    def _on_joint_command(self, msg: JointState) -> None:
        with self._lock:
            self._upper_body.update(zip(msg.name, msg.position))

    def velocity_command(self) -> tuple[float, float, float]:
        with self._lock:
            if time.monotonic() - self._velocity_time > COMMAND_TIMEOUT_S:
                return (0.0, 0.0, 0.0)
            return self._velocity

    def upper_body_targets(self) -> dict[str, float]:
        with self._lock:
            return dict(self._upper_body)

    def publish(
        self,
        sim_time_s: float,
        joint_names: Sequence[str],
        joint_positions: Sequence[float],
        base_position: Sequence[float],
        base_orientation_xyzw: Sequence[float],
        base_linear_velocity: Sequence[float],
        base_angular_velocity: Sequence[float],
    ) -> None:
        seconds = int(sim_time_s)
        stamp = Clock().clock
        stamp.sec, stamp.nanosec = seconds, int((sim_time_s - seconds) * 1e9)
        self._clock_pub.publish(Clock(clock=stamp))

        joints = JointState(name=list(joint_names), position=[float(p) for p in joint_positions])
        joints.header.stamp = stamp
        self._joint_pub.publish(joints)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "pelvis"
        pose = odom.pose.pose
        pose.position.x, pose.position.y, pose.position.z = (float(v) for v in base_position)
        q = pose.orientation
        q.x, q.y, q.z, q.w = (float(v) for v in base_orientation_xyzw)
        twist = odom.twist.twist
        twist.linear.x, twist.linear.y, twist.linear.z = (float(v) for v in base_linear_velocity)
        twist.angular.x, twist.angular.y, twist.angular.z = (float(v) for v in base_angular_velocity)
        self._odom_pub.publish(odom)

    def shutdown(self) -> None:
        self._executor.shutdown(timeout_sec=1.0)
        if getattr(self, "_image_worker", None) is not None:
            self._image_worker.shutdown(wait=True)  # the last image is sent before the node goes
        self._node.destroy_node()
        rclpy.try_shutdown()
