#!/usr/bin/env python3
"""
Test.py

Live evaluation of the trained VIO model against Gazebo.

It connects to the running simulation, flies the drone along a chosen path,
feeds the live camera and IMU stream through the model, integrates the
predicted relative poses into a full trajectory, and plots that estimated
trajectory against the ground truth in a matplotlib window that updates while
the drone is flying.

Run (with the simulation already up):
    python3 Test.py --checkpoint vio_model.pt --path square
    python3 Test.py --checkpoint vio_model.pt --path circle
    python3 Test.py --no-fly            # you drive the drone yourself

The model, preprocessing and pose math are imported from Train.py and
DataLoader.py so they exactly match training.
"""

import time
import argparse
import threading
from collections import deque

import numpy as np

import torch

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Empty, Bool

from cv_bridge import CvBridge

import matplotlib.pyplot as plt

from DataLoader import (
    build_image_transform,
    bgr_to_input_tensor,
    resample_imu,
    normalize_imu,
    quat_to_matrix,
    integrate_pose,
)
from Train import VIONet


# ---------------------------------------------------------------------------
# Flight controller. Reuse the collector's controller if it is importable,
# otherwise fall back to a small built in one so Test.py can stand alone.
# ---------------------------------------------------------------------------

try:
    from collect_vio_dataset import DroneController, build_paths
    _HAVE_COLLECTOR = True
except Exception:
    _HAVE_COLLECTOR = False
    import math

    class DroneController(Node):
        def __init__(self):
            super().__init__("vio_test_controller")
            self.cmd_pub = self.create_publisher(Twist, "/simple_drone/cmd_vel", 10)
            self.takeoff_pub = self.create_publisher(Empty, "/simple_drone/takeoff", 10)
            self.land_pub = self.create_publisher(Empty, "/simple_drone/land", 10)
            self.reset_pub = self.create_publisher(Empty, "/simple_drone/reset", 10)
            self.vel_mode_pub = self.create_publisher(Bool, "/simple_drone/dronevel_mode", 10)
            self.posctrl_pub = self.create_publisher(Bool, "/simple_drone/posctrl", 10)

        def publish_velocity(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
            m = Twist()
            m.linear.x, m.linear.y, m.linear.z, m.angular.z = float(x), float(y), float(z), float(yaw)
            self.cmd_pub.publish(m)

        def hover(self):
            self.publish_velocity()

        def _flag(self, pub, val):
            m = Bool()
            m.data = val
            pub.publish(m)

        def prepare(self, climb_speed=0.8, climb_time=3.0):
            self.reset_pub.publish(Empty())
            time.sleep(2.0)
            end = time.monotonic() + 2.0
            while time.monotonic() < end and rclpy.ok():
                self.takeoff_pub.publish(Empty())
                self._flag(self.vel_mode_pub, True)
                self._flag(self.posctrl_pub, False)
                self.hover()
                time.sleep(0.1)
            end = time.monotonic() + climb_time
            while time.monotonic() < end and rclpy.ok():
                self.publish_velocity(0.0, 0.0, climb_speed, 0.0)
                time.sleep(0.05)
            self.hover()
            time.sleep(0.5)

        def fly_path(self, path, rate_hz=20.0):
            period = 1.0 / rate_hz
            start = time.monotonic()
            while rclpy.ok():
                t = time.monotonic() - start
                if t >= path["duration"]:
                    break
                vx, vy, vz, yaw = path["vel"](t)
                self.publish_velocity(vx, vy, vz, yaw)
                time.sleep(period)
            self.hover()

        def settle_and_land(self):
            self.hover()
            time.sleep(1.0)
            self.land_pub.publish(Empty())
            time.sleep(2.0)

    def build_paths():
        omega = 0.4
        loop = 2 * math.pi / omega
        v, side = 0.6, 5.0
        return {
            "square": {"vel": lambda t: (
                (v, 0, 0, 0) if t < side else
                (0, v, 0, 0) if t < 2 * side else
                (-v, 0, 0, 0) if t < 3 * side else
                (0, -v, 0, 0)), "duration": 4 * side},
            "circle": {"vel": lambda t: (0.5, 0.0, 0.0, omega), "duration": loop},
        }


# ---------------------------------------------------------------------------
# ROS node that only buffers sensor data for the inference loop
# ---------------------------------------------------------------------------

class VIOSensorNode(Node):
    def __init__(self, camera, image_size):
        super().__init__("vio_test_sensors")
        self.bridge = CvBridge()
        self.transform = build_image_transform(image_size)
        self.lock = threading.Lock()

        self.frame_buf = deque()               # (ts, bgr_image)
        self.imu_buf = deque(maxlen=20000)     # (ts, np6)
        self.latest_gt = None                  # (pos3, quat4)

        image_topic = f"/simple_drone/{camera}/image_raw"
        self.create_subscription(Image, image_topic, self.image_cb, 10)
        self.create_subscription(Imu, "/simple_drone/imu/out", self.imu_cb, 100)
        self.create_subscription(Pose, "/simple_drone/gt_pose", self.pose_cb, 50)

    def image_cb(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.frame_buf.append((ts, frame))

    def imu_cb(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        vec = np.array([
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
        ], dtype=np.float32)
        with self.lock:
            self.imu_buf.append((ts, vec))

    def pose_cb(self, msg):
        with self.lock:
            self.latest_gt = (
                np.array([msg.position.x, msg.position.y, msg.position.z], dtype=np.float64),
                np.array([msg.orientation.x, msg.orientation.y,
                          msg.orientation.z, msg.orientation.w], dtype=np.float64),
            )

    def pop_frames(self):
        with self.lock:
            frames = list(self.frame_buf)
            self.frame_buf.clear()
        return frames

    def imu_between(self, t0, t1):
        with self.lock:
            if not self.imu_buf:
                return np.zeros((0, 6), dtype=np.float32)
            ts = np.array([x[0] for x in self.imu_buf])
            data = np.stack([x[1] for x in self.imu_buf], axis=0)
        mask = (ts >= t0) & (ts <= t1)
        return data[mask]

    def get_gt(self):
        with self.lock:
            return self.latest_gt


# ---------------------------------------------------------------------------
# Live plot
# ---------------------------------------------------------------------------

class LivePlot:
    def __init__(self):
        plt.ion()
        self.fig, (self.ax_xy, self.ax_z) = plt.subplots(1, 2, figsize=(12, 5))

        self.gt_xy, = self.ax_xy.plot([], [], "g-", label="ground truth")
        self.est_xy, = self.ax_xy.plot([], [], "r-", label="VIO estimate")
        self.ax_xy.set_title("Top down trajectory")
        self.ax_xy.set_xlabel("x (m)")
        self.ax_xy.set_ylabel("y (m)")
        self.ax_xy.axis("equal")
        self.ax_xy.grid(True)
        self.ax_xy.legend()

        self.gt_z, = self.ax_z.plot([], [], "g-", label="ground truth")
        self.est_z, = self.ax_z.plot([], [], "r-", label="VIO estimate")
        self.ax_z.set_title("Altitude over steps")
        self.ax_z.set_xlabel("step")
        self.ax_z.set_ylabel("z (m)")
        self.ax_z.grid(True)
        self.ax_z.legend()

    def update(self, gt, est, rmse):
        gt = np.array(gt)
        est = np.array(est)
        if len(gt):
            self.gt_xy.set_data(gt[:, 0], gt[:, 1])
            self.gt_z.set_data(range(len(gt)), gt[:, 2])
        if len(est):
            self.est_xy.set_data(est[:, 0], est[:, 1])
            self.est_z.set_data(range(len(est)), est[:, 2])
        for ax in (self.ax_xy, self.ax_z):
            ax.relim()
            ax.autoscale_view()
        self.fig.suptitle(f"position RMSE: {rmse:.3f} m")
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def keep_open(self):
        plt.ioff()
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = VIONet(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt["imu_mean"], ckpt["imu_std"]


def main():
    parser = argparse.ArgumentParser(description="Live VIO test against Gazebo.")
    parser.add_argument("--checkpoint", default="vio_model.pt")
    parser.add_argument("--path", default=None, help="trajectory to fly (default: config test_path)")
    parser.add_argument("--no-fly", action="store_true", help="do not command the drone, just observe")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, imu_mean, imu_std = load_model(args.checkpoint, device)
    N = cfg["stack_size"]
    L = cfg["imu_seq_len"]
    print(f"Loaded model on {device}. stack_size={N}, imu_seq_len={L}")

    paths = build_paths()
    path_name = args.path or cfg.get("test_path", "square")
    if not args.no_fly and path_name not in paths:
        raise SystemExit(f"Unknown path '{path_name}'. Options: {list(paths.keys())}")

    rclpy.init()
    sensors = VIOSensorNode(cfg["camera"], cfg["image_size"])
    controller = DroneController()

    executor = SingleThreadedExecutor()
    executor.add_node(sensors)
    executor.add_node(controller)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    flight_done = threading.Event()

    def fly():
        if args.no_fly:
            print("no-fly mode: waiting while you drive the drone. Ctrl+C to stop.")
            try:
                while rclpy.ok():
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
        else:
            print(f"Flying path '{path_name}'")
            controller.prepare()
            controller.fly_path(paths[path_name], rate_hz=cfg["test_control_rate"])
            controller.settle_and_land()
            print("Flight finished")
        flight_done.set()

    flight_thread = threading.Thread(target=fly, daemon=True)
    flight_thread.start()

    # inference state
    plot = LivePlot()
    window = []                 # list of (ts, bgr)
    R_est, p_est = None, None
    est_traj, gt_traj = [], []
    transform = sensors.transform

    def preprocess_window(win):
        imgs = [bgr_to_input_tensor(bgr, transform) for _, bgr in win]
        images = torch.stack(imgs, dim=0).unsqueeze(0).to(device)   # (1, N, 3, H, W)
        block = sensors.imu_between(win[0][0], win[-1][0])
        imu_seq = normalize_imu(resample_imu(block, L), imu_mean, imu_std)
        imu = torch.from_numpy(imu_seq).unsqueeze(0).to(device)     # (1, L, 6)
        return images, imu

    print("Waiting for frames...")
    try:
        while rclpy.ok():
            for ts, bgr in sensors.pop_frames():
                # initialise the estimate on the very first frame we see
                if R_est is None:
                    gt = sensors.get_gt()
                    if gt is None:
                        continue
                    p_est = gt[0].copy()
                    R_est = quat_to_matrix(*gt[1])
                    est_traj.append(p_est.copy())
                    gt_traj.append(gt[0].copy())

                window.append((ts, bgr))
                if len(window) == N:
                    images, imu = preprocess_window(window)
                    with torch.no_grad():
                        pred = model(images, imu).cpu().numpy()[0]
                    R_est, p_est = integrate_pose(R_est, p_est, pred)

                    est_traj.append(p_est.copy())
                    gt = sensors.get_gt()
                    gt_traj.append(gt[0].copy() if gt is not None else gt_traj[-1])

                    window = [window[-1]]   # share the boundary frame

            # running error and live plot
            m = min(len(est_traj), len(gt_traj))
            rmse = 0.0
            if m > 1:
                diff = np.array(est_traj[:m]) - np.array(gt_traj[:m])
                rmse = float(np.sqrt((diff ** 2).sum(axis=1).mean()))
            plot.update(gt_traj, est_traj, rmse)

            if flight_done.is_set() and not sensors.frame_buf:
                break
    except KeyboardInterrupt:
        print("Stopped.")

    # final report
    m = min(len(est_traj), len(gt_traj))
    if m > 1:
        diff = np.array(est_traj[:m]) - np.array(gt_traj[:m])
        rmse = float(np.sqrt((diff ** 2).sum(axis=1).mean()))
        print(f"Final position RMSE over {m} steps: {rmse:.3f} m")
    plot.update(gt_traj, est_traj, rmse if m > 1 else 0.0)

    executor.shutdown()
    sensors.destroy_node()
    controller.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    print("Close the plot window to exit.")
    plot.keep_open()


if __name__ == "__main__":
    main()