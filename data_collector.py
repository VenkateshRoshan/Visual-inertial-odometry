#!/usr/bin/env python3
"""
VIO dataset collector for the simple_drone Gazebo model.

What this script does:
  * Defines a dictionary of flight paths (square, circle, spiral, helix, ...).
  * Flies the drone along each selected path, one at a time.
  * Records synchronised camera, IMU and ground truth data for each path.
  * Stores every path in its own folder inside a per run folder.
  * Keeps a single global frame counter so image ids never repeat across
    runs or worlds. The counter can be forced with --start-count, or it is
    auto detected by scanning frames that already exist under --output-dir.

Two classes do the work:
  * DroneController: publishes takeoff, mode and velocity commands.
  * DataCollector:   subscribes to the sensors and writes the CSV and images.

Run it while the simulation is up, for example:
  python3 collect_vio_dataset.py --output-dir vio_dataset --run-name warehouse
  python3 collect_vio_dataset.py --output-dir vio_dataset --run-name forest
  python3 collect_vio_dataset.py --list-paths
  python3 collect_vio_dataset.py --paths square,circle,spiral
"""

import os
import csv
import time
import math
import json
import argparse
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Empty, Bool

from cv_bridge import CvBridge
import cv2


# ---------------------------------------------------------------------------
# Path library
# ---------------------------------------------------------------------------
#
# Each path is a dict with:
#   "vel":      a function of elapsed time t that returns (vx, vy, vz, yaw)
#   "duration": how long to fly that path, in seconds
#
# Linear values are body frame velocities in m/s, yaw is the yaw rate in rad/s.
# A steady forward velocity combined with a steady yaw rate traces a circle,
# which is why circle, spiral, helix and figure eight are built that way.


def _segments(seg_list):
    """Build a path from a list of (duration, vx, vy, vz, yaw) segments."""
    total = sum(s[0] for s in seg_list)

    def vel(t):
        acc = 0.0
        for dur, vx, vy, vz, yaw in seg_list:
            if t < acc + dur:
                return vx, vy, vz, yaw
            acc += dur
        return 0.0, 0.0, 0.0, 0.0

    return {"vel": vel, "duration": total}


def _const(vx, vy, vz, yaw, duration):
    """A single steady velocity command held for the whole duration."""
    return {"vel": lambda t: (vx, vy, vz, yaw), "duration": duration}


def build_paths():
    paths = {}

    # 1. straight line: forward then back
    paths["line"] = _segments([
        (6.0,  0.6, 0.0, 0.0, 0.0),
        (6.0, -0.6, 0.0, 0.0, 0.0),
    ])

    # 2. square: four body frame sides
    v, side = 0.6, 5.0
    paths["square"] = _segments([
        (side,  v,   0.0, 0.0, 0.0),
        (side,  0.0, v,   0.0, 0.0),
        (side, -v,   0.0, 0.0, 0.0),
        (side,  0.0, -v,  0.0, 0.0),
    ])

    # 3. triangle: three forward legs with 120 degree yaw turns between them
    yaw_r = 0.6
    turn120 = math.radians(120) / yaw_r
    paths["triangle"] = _segments([
        (5.0,     0.6, 0.0, 0.0, 0.0),
        (turn120, 0.0, 0.0, 0.0, yaw_r),
        (5.0,     0.6, 0.0, 0.0, 0.0),
        (turn120, 0.0, 0.0, 0.0, yaw_r),
        (5.0,     0.6, 0.0, 0.0, 0.0),
        (turn120, 0.0, 0.0, 0.0, yaw_r),
    ])

    # 4. circle: steady forward plus steady yaw = one full loop
    omega = 0.4
    loop = 2 * math.pi / omega
    paths["circle"] = _const(0.5, 0.0, 0.0, omega, loop)

    # 5. figure eight: one loop each direction
    def fig8(t):
        return (0.5, 0.0, 0.0, omega) if t < loop else (0.5, 0.0, 0.0, -omega)
    paths["figure_eight"] = {"vel": fig8, "duration": 2 * loop}

    # 6. horizontal spiral: yaw rate decays so the radius keeps growing
    def spiral(t):
        w = 0.7 / (1.0 + 0.05 * t)
        return 0.5, 0.0, 0.0, w
    paths["spiral"] = {"vel": spiral, "duration": 35.0}

    # 7. helix: circle while climbing, then circle while descending
    def helix(t):
        vz = 0.2 if t < 15.0 else -0.2
        return 0.5, 0.0, vz, omega
    paths["helix"] = {"vel": helix, "duration": 30.0}

    # 8. sine wave: forward with a smooth lateral sway
    def sine(t):
        vy = 0.5 * math.sin(2 * math.pi * 0.15 * t)
        return 0.6, vy, 0.0, 0.0
    paths["sine_wave"] = {"vel": sine, "duration": 24.0}

    # 9. zigzag: forward with sharp lateral switches
    def zigzag(t):
        vy = 0.5 if int(t // 2.0) % 2 == 0 else -0.5
        return 0.5, vy, 0.0, 0.0
    paths["zigzag"] = {"vel": zigzag, "duration": 24.0}

    # 10. staircase: forward with alternating climb and hold
    def stairs(t):
        vz = 0.4 if int(t // 3.0) % 2 == 0 else 0.0
        return 0.4, 0.0, vz, 0.0
    paths["staircase"] = {"vel": stairs, "duration": 24.0}

    # 11. vertical bob: pure up and down, no translation
    paths["vertical"] = _segments([
        (5.0, 0.0, 0.0,  0.5, 0.0),
        (5.0, 0.0, 0.0, -0.5, 0.0),
        (5.0, 0.0, 0.0,  0.5, 0.0),
        (5.0, 0.0, 0.0, -0.5, 0.0),
    ])

    # 12. yaw sweep: hover and spin a full turn each way (rotation only)
    yaw_rate = 0.5
    full_turn = 2 * math.pi / yaw_rate
    paths["yaw_sweep"] = _segments([
        (full_turn, 0.0, 0.0, 0.0,  yaw_rate),
        (full_turn, 0.0, 0.0, 0.0, -yaw_rate),
    ])

    # 13. lawnmower survey: long legs joined by 90 degree turns
    turn90 = math.radians(90) / yaw_r
    paths["lawnmower"] = _segments([
        (5.0,    0.5, 0.0, 0.0, 0.0),
        (turn90, 0.0, 0.0, 0.0, yaw_r),
        (2.0,    0.5, 0.0, 0.0, 0.0),
        (turn90, 0.0, 0.0, 0.0, yaw_r),
        (5.0,    0.5, 0.0, 0.0, 0.0),
        (turn90, 0.0, 0.0, 0.0, -yaw_r),
        (2.0,    0.5, 0.0, 0.0, 0.0),
        (turn90, 0.0, 0.0, 0.0, -yaw_r),
        (5.0,    0.5, 0.0, 0.0, 0.0),
    ])

    return paths


# ---------------------------------------------------------------------------
# Math helper
# ---------------------------------------------------------------------------

def quaternion_to_euler(x, y, z, w):
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = max(min(t2, 1.0), -1.0)
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Class 1: drives the drone
# ---------------------------------------------------------------------------

class DroneController(Node):
    """Publishes takeoff, mode, reset and velocity commands."""

    def __init__(self):
        super().__init__("vio_drone_controller")

        self.cmd_pub = self.create_publisher(Twist, "/simple_drone/cmd_vel", 10)
        self.takeoff_pub = self.create_publisher(Empty, "/simple_drone/takeoff", 10)
        self.land_pub = self.create_publisher(Empty, "/simple_drone/land", 10)
        self.reset_pub = self.create_publisher(Empty, "/simple_drone/reset", 10)
        self.vel_mode_pub = self.create_publisher(Bool, "/simple_drone/dronevel_mode", 10)
        self.posctrl_pub = self.create_publisher(Bool, "/simple_drone/posctrl", 10)

    def publish_velocity(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        msg = Twist()
        msg.linear.x = float(x)
        msg.linear.y = float(y)
        msg.linear.z = float(z)
        msg.angular.z = float(yaw)
        self.cmd_pub.publish(msg)

    def hover(self):
        self.publish_velocity(0.0, 0.0, 0.0, 0.0)

    def _flag(self, publisher, value):
        msg = Bool()
        msg.data = value
        publisher.publish(msg)

    def prepare(self, climb_speed=0.8, climb_time=3.0):
        """Reset to origin, take off, enable velocity mode, climb a little."""
        self.get_logger().info("Resetting drone")
        self.reset_pub.publish(Empty())
        time.sleep(2.0)

        self.get_logger().info("Taking off")
        end = time.monotonic() + 2.0
        while time.monotonic() < end and rclpy.ok():
            self.takeoff_pub.publish(Empty())
            self._flag(self.vel_mode_pub, True)
            self._flag(self.posctrl_pub, False)
            self.hover()
            time.sleep(0.1)

        # climb to a working altitude before the path starts
        end = time.monotonic() + climb_time
        while time.monotonic() < end and rclpy.ok():
            self.publish_velocity(0.0, 0.0, climb_speed, 0.0)
            time.sleep(0.05)
        self.hover()
        time.sleep(0.5)

    def fly_path(self, path, rate_hz=20.0):
        """Fly one path by stepping its velocity function until the duration ends."""
        vel = path["vel"]
        duration = path["duration"]
        period = 1.0 / rate_hz

        start = time.monotonic()
        while rclpy.ok():
            t = time.monotonic() - start
            if t >= duration:
                break
            vx, vy, vz, yaw = vel(t)
            self.publish_velocity(vx, vy, vz, yaw)
            time.sleep(period)
        self.hover()

    def settle_and_land(self):
        """Hover briefly then land, so the next path starts clean."""
        self.hover()
        time.sleep(1.0)
        self.land_pub.publish(Empty())
        time.sleep(2.0)


# ---------------------------------------------------------------------------
# Class 2: records the data
# ---------------------------------------------------------------------------

class DataCollector(Node):
    """Subscribes to camera, IMU and ground truth, and writes them to disk."""

    def __init__(self, run_root, start_count, camera="bottom"):
        super().__init__("vio_data_collector")

        self.bridge = CvBridge()
        self.run_root = run_root
        self.frame_id = start_count          # global, never resets between paths
        self.lock = threading.Lock()

        # per path state, filled in by start_path
        self.active = False
        self.path_name = None
        self.image_dir = None
        self.gt_file = None
        self.imu_file = None
        self.samples_file = None
        self.gt_writer = None
        self.imu_writer = None
        self.samples_writer = None
        self.path_image_count = 0
        self.imu_row_index = 0
        self.imu_idx_at_last_image = 0
        self.imu_window = deque()

        # latest ground truth, shared between callbacks
        self.latest_pose = None
        self.latest_pose_time = None
        self.latest_vel = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        image_topic = f"/simple_drone/{camera}/image_raw"
        self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.create_subscription(Imu, "/simple_drone/imu/out", self.imu_callback, 100)
        self.create_subscription(Pose, "/simple_drone/gt_pose", self.pose_callback, 50)
        self.create_subscription(Twist, "/simple_drone/gt_vel", self.vel_callback, 50)

        self.get_logger().info(f"Collector ready, images start at id {start_count}")

    # -- helpers -----------------------------------------------------------

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _ros_time(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    # -- path lifecycle ----------------------------------------------------

    def start_path(self, path_name):
        """Open fresh folders and CSV files for a new path."""
        with self.lock:
            path_dir = os.path.join(self.run_root, path_name)
            self.image_dir = os.path.join(path_dir, "images")
            os.makedirs(self.image_dir, exist_ok=True)

            self.gt_file = open(os.path.join(path_dir, "groundtruth.csv"), "w", newline="")
            self.imu_file = open(os.path.join(path_dir, "imu.csv"), "w", newline="")
            self.samples_file = open(os.path.join(path_dir, "samples.csv"), "w", newline="")

            self.gt_writer = csv.writer(self.gt_file)
            self.imu_writer = csv.writer(self.imu_file)
            self.samples_writer = csv.writer(self.samples_file)

            self.gt_writer.writerow([
                "timestamp", "x", "y", "z",
                "qx", "qy", "qz", "qw",
                "roll", "pitch", "yaw",
                "vx", "vy", "vz", "wx", "wy", "wz",
            ])
            self.imu_writer.writerow([
                "timestamp", "ax", "ay", "az",
                "gx", "gy", "gz", "qx", "qy", "qz", "qw",
            ])
            self.samples_writer.writerow([
                "image_id", "image_timestamp", "image_path",
                "pose_timestamp", "x", "y", "z", "qx", "qy", "qz", "qw",
                "imu_start_timestamp", "imu_end_timestamp",
                "imu_start_index", "imu_end_index",
            ])

            self.path_name = path_name
            self.path_image_count = 0
            self.imu_row_index = 0
            self.imu_idx_at_last_image = 0
            self.imu_window.clear()
            self.active = True

        self.get_logger().info(f"Started path '{path_name}'")

    def stop_path(self):
        """Close CSV files and report how many images this path captured."""
        with self.lock:
            self.active = False
            for f in (self.gt_file, self.imu_file, self.samples_file):
                if f is not None:
                    f.close()
            self.gt_file = self.imu_file = self.samples_file = None
            count = self.path_image_count
            name = self.path_name

        self.get_logger().info(f"Finished path '{name}', saved {count} images")
        return count

    # -- callbacks ---------------------------------------------------------

    def pose_callback(self, msg):
        roll, pitch, yaw = quaternion_to_euler(
            msg.orientation.x, msg.orientation.y,
            msg.orientation.z, msg.orientation.w
        )
        with self.lock:
            self.latest_pose = {
                "x": msg.position.x, "y": msg.position.y, "z": msg.position.z,
                "qx": msg.orientation.x, "qy": msg.orientation.y,
                "qz": msg.orientation.z, "qw": msg.orientation.w,
                "roll": roll, "pitch": pitch, "yaw": yaw,
            }
            self.latest_pose_time = self._now()

            if not self.active:
                return
            vx, vy, vz, wx, wy, wz = self.latest_vel
            self.gt_writer.writerow([
                self.latest_pose_time,
                msg.position.x, msg.position.y, msg.position.z,
                msg.orientation.x, msg.orientation.y,
                msg.orientation.z, msg.orientation.w,
                roll, pitch, yaw,
                vx, vy, vz, wx, wy, wz,
            ])

    def vel_callback(self, msg):
        with self.lock:
            self.latest_vel = (
                msg.linear.x, msg.linear.y, msg.linear.z,
                msg.angular.x, msg.angular.y, msg.angular.z,
            )

    def imu_callback(self, msg):
        with self.lock:
            if not self.active:
                return
            ts = self._ros_time(msg.header.stamp)
            self.imu_writer.writerow([
                ts,
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
            ])
            self.imu_window.append((self.imu_row_index, ts))
            self.imu_row_index += 1

    def image_callback(self, msg):
        with self.lock:
            if not self.active or self.latest_pose is None:
                return

            ts = self._ros_time(msg.header.stamp)
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            image_name = f"frame_{self.frame_id:06d}.png"
            image_path = os.path.join(self.image_dir, image_name)
            cv2.imwrite(image_path, frame)

            # image path relative to the run root, so the dataset stays portable
            rel_path = os.path.relpath(image_path, self.run_root)

            if self.imu_window:
                imu_start_idx, imu_start_ts = self.imu_window[0]
                imu_end_idx, imu_end_ts = self.imu_window[-1]
            else:
                imu_start_idx = imu_end_idx = self.imu_idx_at_last_image
                imu_start_ts = imu_end_ts = ""

            p = self.latest_pose
            self.samples_writer.writerow([
                self.frame_id, ts, rel_path,
                self.latest_pose_time,
                p["x"], p["y"], p["z"],
                p["qx"], p["qy"], p["qz"], p["qw"],
                imu_start_ts, imu_end_ts,
                imu_start_idx, imu_end_idx,
            ])

            self.imu_idx_at_last_image = self.imu_row_index
            self.imu_window.clear()

            self.frame_id += 1
            self.path_image_count += 1
            if self.path_image_count % 50 == 0:
                self.get_logger().info(
                    f"[{self.path_name}] saved {self.path_image_count} images"
                )


# ---------------------------------------------------------------------------
# Continuous counter helpers
# ---------------------------------------------------------------------------

def scan_next_frame_id(output_dir):
    """Look at every frame_XXXXXX.png under output_dir and return max id plus one."""
    max_id = -1
    if os.path.isdir(output_dir):
        for dirpath, _, files in os.walk(output_dir):
            for name in files:
                if name.startswith("frame_") and name.endswith(".png"):
                    try:
                        num = int(name[len("frame_"):-len(".png")])
                        max_id = max(max_id, num)
                    except ValueError:
                        pass
    return max_id + 1


def resolve_start_count(output_dir, arg_start):
    """Argument wins if given, otherwise auto detect by scanning existing frames."""
    if arg_start is not None:
        return arg_start

    counter_file = os.path.join(output_dir, "frame_counter.txt")
    scanned = scan_next_frame_id(output_dir)

    from_file = 0
    if os.path.isfile(counter_file):
        try:
            with open(counter_file) as f:
                from_file = int(f.read().strip())
        except (ValueError, OSError):
            from_file = 0

    return max(scanned, from_file)


def write_counter_file(output_dir, next_id):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "frame_counter.txt"), "w") as f:
        f.write(str(next_id))


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    all_paths = build_paths()

    parser = argparse.ArgumentParser(description="Collect a VIO dataset by flying preset paths.")
    parser.add_argument("--output-dir", default="vio_dataset",
                        help="Root folder for all runs. The frame counter lives here.")
    parser.add_argument("--run-name", default=None,
                        help="Name for this run or world, becomes a subfolder. "
                             "Defaults to a timestamp.")
    parser.add_argument("--start-count", type=int, default=None,
                        help="Force the first image id. If omitted it is auto detected.")
    parser.add_argument("--paths", default="all",
                        help="Comma separated path names, or 'all'.")
    parser.add_argument("--camera", default="bottom", choices=["bottom", "front"],
                        help="Which camera to record.")
    parser.add_argument("--rate", type=float, default=20.0,
                        help="Velocity command rate in Hz.")
    parser.add_argument("--list-paths", action="store_true",
                        help="Print the available paths and exit.")
    args = parser.parse_args()

    if args.list_paths:
        print("Available paths:")
        for name, p in all_paths.items():
            print(f"  {name:<14} duration {p['duration']:.1f} s")
        return

    # decide which paths to fly
    if args.paths == "all":
        selected = list(all_paths.keys())
    else:
        selected = [p.strip() for p in args.paths.split(",") if p.strip()]
        unknown = [p for p in selected if p not in all_paths]
        if unknown:
            raise SystemExit(f"Unknown path names: {unknown}")

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_root = os.path.join(args.output_dir, run_name)
    os.makedirs(run_root, exist_ok=True)

    start_count = resolve_start_count(args.output_dir, args.start_count)
    print(f"Run folder : {run_root}")
    print(f"Paths      : {selected}")
    print(f"Start id   : {start_count}")

    rclpy.init()
    controller = DroneController()
    collector = DataCollector(run_root, start_count, camera=args.camera)

    # spin both nodes on a background executor while main does the sequencing
    executor = SingleThreadedExecutor()
    executor.add_node(controller)
    executor.add_node(collector)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    summary = []
    try:
        for name in selected:
            print(f"\n=== Path: {name} ===")
            controller.prepare()
            collector.start_path(name)
            controller.fly_path(all_paths[name], rate_hz=args.rate)
            count = collector.stop_path()
            controller.settle_and_land()
            summary.append({"path": name, "images": count})
    except KeyboardInterrupt:
        print("\nInterrupted, stopping.")
        controller.hover()
        if collector.active:
            collector.stop_path()
    finally:
        next_id = collector.frame_id
        write_counter_file(args.output_dir, next_id)

        # small manifest describing this run
        with open(os.path.join(run_root, "run_info.json"), "w") as f:
            json.dump({
                "run_name": run_name,
                "camera": args.camera,
                "start_count": start_count,
                "next_count": next_id,
                "paths": summary,
            }, f, indent=2)

        print(f"\nDone. Next image id is {next_id} (saved to frame_counter.txt).")
        print(f"Run info written to {os.path.join(run_root, 'run_info.json')}")

        executor.shutdown()
        controller.destroy_node()
        collector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()