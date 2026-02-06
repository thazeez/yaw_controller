#!/usr/bin/env python3
"""
python3 YawController_OptiWorld_Clean.py \
  --pose-topic /Drone/pose \
  --yaw-deg 90 \
  --kp-yaw 1.0 \
  --yaw-rate-max 0.3 \
  --tol-deg 2.0 \
  --yaw-sign -1 \
  --warmup 1.0 --hz 50 \
  --logfile /home/root/yaw_log_opti_abs90.csv
"""

"""
YawController_OptiWorld_Clean.py

Yaw-only P controller using OptiTrack pose (/Drone/pose):
- Uses OptiTrack PoseStamped orientation (ROS quaternion xyzw)
- Computes yaw about +Z (Z-up world)
- ✅ Supports Opti yaw sign convention: you said "orientation Z reduces CCW"
  -> set --yaw-sign -1 (default) to flip yaw so CCW becomes positive in our math

Publishes:
- /cmd_vel (geometry_msgs/Twist)
  - linear.x/y/z = 0
  - angular.z = yaw rate command (rad/s)

Modes (choose ONE):
A) Absolute world yaw:
   --yaw-deg 90

B) Relative world yaw step from latched yaw:
   --yaw-step-deg 90

C) Face a goal point in OptiTrack world:
   --face-goal --x 1.3 --y -1.0

Logging:
CSV columns:
t, phase, x, y, z, yaw_deg, yaw_des_deg, yaw_err_deg, yaw_rate_cmd

Run examples at bottom.
"""

import csv
import time
import math
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist, PoseStamped


def qos_reliable(depth=10):
    q = QoSProfile(depth=depth)
    q.history = HistoryPolicy.KEEP_LAST
    q.reliability = ReliabilityPolicy.RELIABLE
    q.durability = DurabilityPolicy.VOLATILE
    return q


def qos_best_effort(depth=10):
    q = QoSProfile(depth=depth)
    q.history = HistoryPolicy.KEEP_LAST
    q.reliability = ReliabilityPolicy.BEST_EFFORT
    q.durability = DurabilityPolicy.VOLATILE
    return q


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat_xyzw(x, y, z, w):
    """
    Yaw about +Z from quaternion (ROS xyzw) for a Z-up world.
    Standard formula (assumes quaternion represents body orientation in world).
    """
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(s, c)


class YawControllerOptiWorldClean(Node):
    def __init__(self, args):
        super().__init__("yaw_controller_optiworld_clean")

        self.pose_topic = args.pose_topic
        self.kp_yaw = float(args.kp_yaw)
        self.yaw_rate_max = float(args.yaw_rate_max)
        self.tol_rad = math.radians(float(args.tol_deg))

        self.hz = float(args.hz)
        self.dt = 1.0 / self.hz
        self.warmup_s = float(args.warmup)

        # ✅ yaw sign handling (your Opti: "yaw reduces CCW" => use -1)
        self.yaw_sign = float(args.yaw_sign)

        # goal-facing option
        self.face_goal = bool(args.face_goal)
        self.x_goal = float(args.x)
        self.y_goal = float(args.y)

        # target selection
        self.yaw_target_abs = None
        self.yaw_step_rad = None
        if args.yaw_deg is not None:
            self.yaw_target_abs = math.radians(float(args.yaw_deg))
        if args.yaw_step_deg is not None:
            self.yaw_step_rad = math.radians(float(args.yaw_step_deg))

        if not self.face_goal:
            if (self.yaw_target_abs is None) and (self.yaw_step_rad is None):
                raise ValueError("Provide either --yaw-deg or --yaw-step-deg, OR use --face-goal with --x --y.")

        # state from OptiTrack
        self.have_pose = False
        self.px = 0.0
        self.py = 0.0
        self.pz = 0.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.qw = 1.0

        self.latched = False
        self.yaw0 = 0.0
        self.yaw_des = float("nan")
        self.t_start = time.time()

        # pub/sub
        self.pub = self.create_publisher(Twist, "/cmd_vel", qos_reliable())
        self.create_subscription(PoseStamped, self.pose_topic, self.pose_cb, qos_best_effort())

        # logging
        self.log_enabled = bool(args.logfile)
        self.f = None
        self.w = None
        if self.log_enabled:
            self.f = open(args.logfile, "w", newline="")
            self.w = csv.writer(self.f)
            self.w.writerow([
                "t", "phase",
                "x", "y", "z",
                "yaw_deg", "yaw_des_deg", "yaw_err_deg",
                "yaw_rate_cmd",
            ])
            self.f.flush()

        self.timer = self.create_timer(self.dt, self.loop)

        self.get_logger().info(f"Listening pose: {self.pose_topic}")
        self.get_logger().info(f"yaw_sign={self.yaw_sign} (use -1 for 'yaw reduces CCW')")
        if self.face_goal:
            self.get_logger().info(f"Mode: face-goal (x_goal={self.x_goal:.3f}, y_goal={self.y_goal:.3f})")
        elif self.yaw_step_rad is not None:
            self.get_logger().info(f"Mode: yaw-step ({args.yaw_step_deg} deg from latched yaw)")
        else:
            self.get_logger().info(f"Mode: absolute yaw ({args.yaw_deg} deg world)")

    def pose_cb(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        self.px = float(p.x)
        self.py = float(p.y)
        self.pz = float(p.z)

        # ROS quaternion order: x,y,z,w
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
        n = math.sqrt(w*w + x*x + y*y + z*z)
        if n > 1e-9:
            x, y, z, w = x/n, y/n, z/n, w/n
        self.qx, self.qy, self.qz, self.qw = x, y, z, w

        self.have_pose = True

    def publish_cmd(self, yaw_rate_cmd):
        m = Twist()
        m.linear.x = 0.0
        m.linear.y = 0.0
        m.linear.z = 0.0
        m.angular.x = 0.0
        m.angular.y = 0.0
        m.angular.z = float(yaw_rate_cmd)  # rad/s
        self.pub.publish(m)

    def log_row(self, t, phase, yaw, yaw_des, yaw_err, yaw_rate_cmd):
        if not self.log_enabled:
            return

        def deg_or_nan(a):
            return "nan" if (a is None or not math.isfinite(a)) else f"{math.degrees(a):.3f}"

        self.w.writerow([
            f"{t:.3f}", phase,
            f"{self.px:.4f}", f"{self.py:.4f}", f"{self.pz:.4f}",
            deg_or_nan(yaw),
            deg_or_nan(yaw_des),
            deg_or_nan(yaw_err),
            f"{yaw_rate_cmd:.6f}",
        ])
        self.f.flush()

    def loop(self):
        t = time.time() - self.t_start

        if not self.have_pose:
            self.publish_cmd(0.0)
            return

        # ✅ yaw from Opti quaternion, then apply sign correction
        yaw_raw = yaw_from_quat_xyzw(self.qx, self.qy, self.qz, self.qw)
        yaw = wrap_to_pi(self.yaw_sign * yaw_raw)

        # warmup & latch initial yaw (for step mode)
        if (not self.latched) and (t >= self.warmup_s):
            self.latched = True
            self.yaw0 = yaw

            if self.face_goal:
                # yaw_des computed continuously below
                self.yaw_des = float("nan")
            elif self.yaw_step_rad is not None:
                self.yaw_des = wrap_to_pi(self.yaw0 + self.yaw_step_rad)
            elif self.yaw_target_abs is not None:
                self.yaw_des = wrap_to_pi(self.yaw_target_abs)

        if not self.latched:
            # log warmup with yaw_des = NaN (so plot is not misleading)
            self.publish_cmd(0.0)
            self.log_row(t, "warmup", yaw, float("nan"), float("nan"), 0.0)
            return

        # If facing-goal, override yaw_des continuously from OptiTrack position
        if self.face_goal:
            dx = self.x_goal - self.px
            dy = self.y_goal - self.py
            self.yaw_des = wrap_to_pi(math.atan2(dy, dx))

        # yaw error
        e = wrap_to_pi(self.yaw_des - yaw)

        # stop if within tolerance
        if abs(e) <= self.tol_rad:
            self.publish_cmd(0.0)
            self.log_row(t, "hold", yaw, self.yaw_des, e, 0.0)
            return

        # P yaw rate command
        yaw_rate_cmd = self.kp_yaw * e
        yaw_rate_cmd = clamp(yaw_rate_cmd, -self.yaw_rate_max, self.yaw_rate_max)

        self.publish_cmd(yaw_rate_cmd)
        self.log_row(t, "track", yaw, self.yaw_des, e, yaw_rate_cmd)

    def shutdown(self):
        try:
            self.publish_cmd(0.0)
        except Exception:
            pass
        if self.f is not None:
            self.f.close()


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--pose-topic", default="/Drone/pose")

    # choose ONE:
    p.add_argument("--yaw-deg", type=float, default=None, help="Absolute target yaw in OptiTrack world (deg)")
    p.add_argument("--yaw-step-deg", type=float, default=None, help="Relative yaw step from latched yaw (deg)")
    p.add_argument("--face-goal", action="store_true", help="Yaw to face goal point (x,y) in OptiTrack world")
    p.add_argument("--x", type=float, default=0.0, help="Goal x (only used with --face-goal)")
    p.add_argument("--y", type=float, default=0.0, help="Goal y (only used with --face-goal)")

    p.add_argument("--kp-yaw", type=float, default=1.0)
    p.add_argument("--yaw-rate-max", type=float, default=0.3)
    p.add_argument("--tol-deg", type=float, default=2.0)

    p.add_argument("--yaw-sign", type=float, default=-1.0,
                   help="Multiply measured yaw by this sign. Use -1 if Opti yaw decreases for CCW.")

    p.add_argument("--warmup", type=float, default=1.0)
    p.add_argument("--hz", type=float, default=50.0)

    p.add_argument("--logfile", default="/home/root/yaw_log_opti.csv",
                   help="CSV log path. Set to empty string to disable.")

    args = p.parse_args()

    rclpy.init()
    node = YawControllerOptiWorldClean(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
