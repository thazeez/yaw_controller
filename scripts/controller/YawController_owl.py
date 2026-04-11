#!/usr/bin/env python3

# run this way:
# python3 YawController_owl.py --odom-topic /owl/fmu/out/vehicle_odometry --yaw-step-deg 90 --kp-yaw 1.5 --yaw-rate-max 1.0 --tol-deg 2.0 --warmup 1.0 --hz 50 --logfile /home/root/yaw_log_owl.csv

"""
YawController_owl.py

Yaw-only controller (NO linear motion):
- Subscribes to PX4 odometry: /owl/fmu/out/vehicle_odometry
- Computes current yaw (psi) from quaternion (assumed Body -> NED, same convention you used)
- Commands ONLY angular.z on /cmd_vel (rad/s)
- Linear x/y/z are always 0

Modes:
1) Absolute yaw target:
   --yaw-deg 90   (target yaw in degrees, in NED frame convention)

2) Relative yaw step:
   --yaw-step-deg 30   (turn +30 deg from yaw at latch time)

Notes:
- Publishes geometry_msgs/Twist on /cmd_vel
- Uses wrap_to_pi so shortest-turn direction is used
"""

import csv
import time
import math
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from px4_msgs.msg import VehicleOdometry


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


def yaw_from_quat_body_to_ned(w, x, y, z):
    """
    Compute yaw (psi) from quaternion representing Body -> NED.
    Using ZYX convention on R_nb (body to nav).
    """
    r11 = 1 - 2 * (y * y + z * z)
    r21 = 2 * (x * y + z * w)
    return math.atan2(r21, r11)


class YawController(Node):
    def __init__(self, args):
        super().__init__("yaw_controller_cmdvel")

        self.odom_topic = args.odom_topic
        self.kp_yaw = args.kp_yaw
        self.yaw_rate_max = args.yaw_rate_max
        self.tol_rad = math.radians(args.tol_deg)

        self.hz = args.hz
        self.dt = 1.0 / self.hz
        self.warmup_s = args.warmup

        # target selection
        self.yaw_target_abs = None
        self.yaw_step_rad = None
        if args.yaw_deg is not None:
            self.yaw_target_abs = math.radians(args.yaw_deg)
        if args.yaw_step_deg is not None:
            self.yaw_step_rad = math.radians(args.yaw_step_deg)

        if (self.yaw_target_abs is None) and (self.yaw_step_rad is None):
            raise ValueError("Provide either --yaw-deg (absolute) or --yaw-step-deg (relative).")

        # state
        self.have_odom = False
        self.q_w, self.q_x, self.q_y, self.q_z = 1.0, 0.0, 0.0, 0.0
        self.yaw_rate_meas = 0.0

        self.latched = False
        self.yaw0 = 0.0
        self.yaw_des = 0.0
        self.t_start = time.time()

        # pub/sub
        self.pub = self.create_publisher(Twist, "/owl/cmd_vel", qos_reliable())
        self.create_subscription(VehicleOdometry, self.odom_topic, self.odom_cb, qos_best_effort())

        # logging
        self.log_enabled = bool(args.logfile)
        self.f = None
        self.w = None
        if self.log_enabled:
            self.f = open(args.logfile, "w", newline="")
            self.w = csv.writer(self.f)
            self.w.writerow(["t", "phase", "yaw", "yaw_des", "yaw_err", "yaw_rate_cmd", "yaw_rate_meas"])
            self.f.flush()

        self.timer = self.create_timer(self.dt, self.loop)

        self.get_logger().info(f"Listening odom topic: {self.odom_topic}")

    def odom_cb(self, msg: VehicleOdometry):
        q = msg.q  # [w,x,y,z]
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        n = math.sqrt(w * w + x * x + y * y + z * z)
        if n > 1e-9:
            self.q_w, self.q_x, self.q_y, self.q_z = w / n, x / n, y / n, z / n

        try:
            self.yaw_rate_meas = float(msg.angular_velocity[2])
        except Exception:
            self.yaw_rate_meas = 0.0

        self.have_odom = True

    def publish_cmd(self, yaw_rate_cmd):
        m = Twist()
        m.linear.x = 0.0
        m.linear.y = 0.0
        m.linear.z = 0.0
        m.angular.x = 0.0
        m.angular.y = 0.0
        m.angular.z = float(yaw_rate_cmd)
        self.pub.publish(m)

    def log_row(self, t, phase, yaw, yaw_des, yaw_err, yaw_rate_cmd):
        if not self.log_enabled:
            return
        self.w.writerow([
            f"{t:.3f}", phase,
            f"{yaw:.6f}", f"{yaw_des:.6f}", f"{yaw_err:.6f}",
            f"{yaw_rate_cmd:.6f}", f"{self.yaw_rate_meas:.6f}",
        ])
        self.f.flush()

    def loop(self):
        t = time.time() - self.t_start

        if not self.have_odom:
            self.publish_cmd(0.0)
            return

        yaw = yaw_from_quat_body_to_ned(self.q_w, self.q_x, self.q_y, self.q_z)

        # warmup and latch desired yaw
        if (not self.latched) and (t >= self.warmup_s):
            self.latched = True
            self.yaw0 = yaw
            if self.yaw_step_rad is not None:
                self.yaw_des = wrap_to_pi(self.yaw0 + self.yaw_step_rad)
            else:
                self.yaw_des = wrap_to_pi(self.yaw_target_abs)

        if not self.latched:
            self.publish_cmd(0.0)
            self.log_row(t, "warmup", yaw, 0.0, 0.0, 0.0)
            return

        # yaw error (shortest direction)
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

    # Owl-specific default
    p.add_argument("--odom-topic", default="/owl/fmu/out/vehicle_odometry")

    # choose ONE:
    p.add_argument("--yaw-deg", type=float, default=None, help="Absolute target yaw in degrees (NED convention)")
    p.add_argument("--yaw-step-deg", type=float, default=None, help="Relative yaw step in degrees (+CCW from current)")

    p.add_argument("--kp-yaw", type=float, default=1.5, help="Yaw P gain (rad/s per rad)")
    p.add_argument("--yaw-rate-max", type=float, default=1.0, help="Max yaw rate command (rad/s)")
    p.add_argument("--tol-deg", type=float, default=2.0, help="Stop when |yaw error| <= tol (deg)")

    p.add_argument("--warmup", type=float, default=1.0)
    p.add_argument("--hz", type=float, default=50.0)

    p.add_argument("--logfile", default="/home/root/yaw_log_owl.csv",
                   help="CSV log path. Set to empty string to disable.")

    args = p.parse_args()

    rclpy.init()
    node = YawController(args)
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