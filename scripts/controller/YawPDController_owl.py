#!/usr/bin/env python3
"""
YawPDController_owl.py

HOW TO RUN IN TERMINAL:

python3 YawPDController_owl.py \
  --pose-topic /Drone_owl/pose \
  --odom-topic /owl/fmu/out/vehicle_odometry \
  --x 1.0 --y -1.0 --z 0.7 \
  --vx-body 0.20 \
  --vx-body-max 0.30 \
  --ax-body-max 0.80 \
  --kp-z 0.8 --kd-z 0.1 \
  --kp-yaw 1.0 --yaw-rate-max 0.3 \
  --yaw-sign -1 \
  --xy-tol 0.08 --z-tol 0.05 \
  --slow-radius-xy 0.30 \
  --vz-world-max 0.40 \
  --bz-max 0.50 \
  --warmup 1.0 --hz 50 \
  --logfile /home/root/Yaw_owl.csv


WHAT THIS DOES:
- XY guidance: desired heading is computed from CURRENT position -> desired position
- Yaw control: P controller on heading error
- Forward motion: BODY x velocity command
- BODY y velocity is always zero
- Z control: WORLD z PD controller using OptiTrack z
- BODY z command is solved using PX4 attitude quaternion so the resulting NED vertical
  velocity matches the WORLD z command as closely as possible through your relay

NEW (minimal change):
- Forward BODY x command is speed-limited and acceleration-limited
- Decision uses measured BODY x velocity
- Ramp uses previous commanded BODY x velocity
- yaw_des is frozen when close to goal (inside slow-radius zone)
- hold is sticky once reached

PUBLISHES:
- /owl/cmd_vel (geometry_msgs/Twist)
  linear.x = forward body command (FRD forward)
  linear.y = 0
  linear.z = solved body-z command (FRD down)
  angular.z = yaw-rate command

USE WITH YOUR RELAY IN ANOTHER TERMINAL:
python3 offboard_vel_relay_body3_fullquat_withyaw.py \
  --cmd-timeout 0.3 \
  --offboard-prestream 0.8 \
  --yawrate-max 1.0
"""

import csv
import time
import math
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Twist
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


def yaw_from_quat_xyzw(x, y, z, w):
    # Extract yaw from OptiTrack quaternion (xyzw ordering)
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(s, c)


def quat_to_R_body_to_ned(w, x, y, z):
    """
    Rotation matrix R such that:
        v_ned = R * v_body
    body = PX4 FRD
    ned  = PX4 local NED
    """
    R = [[0.0] * 3 for _ in range(3)]

    R[0][0] = 1.0 - 2.0 * (y * y + z * z)
    R[0][1] = 2.0 * (x * y - z * w)
    R[0][2] = 2.0 * (x * z + y * w)

    R[1][0] = 2.0 * (x * y + z * w)
    R[1][1] = 1.0 - 2.0 * (x * x + z * z)
    R[1][2] = 2.0 * (y * z - x * w)

    R[2][0] = 2.0 * (x * z - y * w)
    R[2][1] = 2.0 * (y * z + x * w)
    R[2][2] = 1.0 - 2.0 * (x * x + y * y)

    return R


def Rt_times_v(R, v):
    """
    v_body = R^T * v_ned
    """
    return [
        R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
        R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
        R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2],
    ]


class YawPDController(Node):
    def __init__(self, args):
        super().__init__("yaw_pd_controller")

        # Topics
        self.pose_topic = args.pose_topic
        self.odom_topic = args.odom_topic

        # Goal
        self.x_des = float(args.x)
        self.y_des = float(args.y)
        self.z_des = float(args.z)

        # Forward command in BODY FRD
        self.vx_body_cmd_const = float(args.vx_body)

        # Body x speed/acceleration limits
        self.vx_body_max = float(args.vx_body_max)
        self.ax_body_max = float(args.ax_body_max)

        # Z controller (WORLD z)
        self.kp_z = float(args.kp_z)
        self.kd_z = float(args.kd_z)
        self.vz_world_max = float(abs(args.vz_world_max))
        self.bz_max = float(abs(args.bz_max))

        # Yaw controller
        self.kp_yaw = float(args.kp_yaw)
        self.yaw_rate_max = float(abs(args.yaw_rate_max))
        self.yaw_sign = float(args.yaw_sign)

        # Tolerances
        self.xy_tol = float(abs(args.xy_tol))
        self.z_tol = float(abs(args.z_tol))
        self.slow_radius_xy = float(abs(args.slow_radius_xy))

        # Timing
        self.warmup_s = float(args.warmup)
        self.hz = float(args.hz)
        self.dt = 1.0 / self.hz

        # ROS
        self.pub = self.create_publisher(Twist, "/owl/cmd_vel", qos_reliable())
        self.create_subscription(PoseStamped, self.pose_topic, self.pose_cb, qos_best_effort())
        self.create_subscription(VehicleOdometry, self.odom_topic, self.odom_cb, qos_best_effort())

        # Pose state (OptiTrack)
        self.have_pose = False
        self.x_w = 0.0
        self.y_w = 0.0
        self.z_w = 0.0
        self.qx_opti = 0.0
        self.qy_opti = 0.0
        self.qz_opti = 0.0
        self.qw_opti = 1.0

        # PX4 odom quaternion for relay/body-z solving
        self.have_odom = False
        self.qw_px4 = 1.0
        self.qx_px4 = 0.0
        self.qy_px4 = 0.0
        self.qz_px4 = 0.0

        # PX4 odom NED velocity + measured body velocity
        self.vN_m = 0.0
        self.vE_m = 0.0
        self.vD_m = 0.0
        self.vx_body_m = 0.0
        self.vy_body_m = 0.0
        self.vz_body_m = 0.0

        # Controller state
        self.latched = False
        self.hold = False
        self.t_start = time.time()
        self.ez_last = 0.0
        self.yaw_des_last = 0.0

        # Previous commanded body x for acceleration limiting
        self.vx_body_prev = 0.0

        # Logging
        self.f = open(args.logfile, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "t", "phase",
            "x", "y", "z",
            "x_des", "y_des", "z_des",
            "ex_xy", "ey_xy", "e_xy_norm",
            "ez", "dez",
            "yaw_deg", "yaw_des_deg", "yaw_err_deg",
            "vx_body_cmd", "vy_body_cmd", "vz_body_cmd",
            "vx_body_raw",
            "vx_body_meas", "vy_body_meas", "vz_body_meas",
            "vz_world_up_raw", "vz_world_up_cmd",
            "vD_des",
            "vz_body_raw",
            "yaw_rate_cmd",
            "R20", "R21", "R22",
        ])
        self.f.flush()

        self.timer = self.create_timer(self.dt, self.loop)

        self.get_logger().info("YawPDController started")
        self.get_logger().info(f"Goal: ({self.x_des:.3f}, {self.y_des:.3f}, {self.z_des:.3f})")
        self.get_logger().info(f"Forward body x command: {self.vx_body_cmd_const:.3f} m/s")
        self.get_logger().info(f"Yaw sign: {self.yaw_sign}")
        self.get_logger().info(f"Listening pose topic: {self.pose_topic}")
        self.get_logger().info(f"Listening odom topic: {self.odom_topic}")
        self.get_logger().info(f"Logging to: {args.logfile}")

    def pose_cb(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation

        self.x_w = float(p.x)
        self.y_w = float(p.y)
        self.z_w = float(p.z)

        x = float(q.x)
        y = float(q.y)
        z = float(q.z)
        w = float(q.w)

        n = math.sqrt(w * w + x * x + y * y + z * z)
        if n > 1e-9:
            x /= n
            y /= n
            z /= n
            w /= n

        self.qx_opti = x
        self.qy_opti = y
        self.qz_opti = z
        self.qw_opti = w

        self.have_pose = True

    def odom_cb(self, msg: VehicleOdometry):
        w, x, y, z = float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3])
        n = math.sqrt(w * w + x * x + y * y + z * z)
        if n > 1e-9:
            w, x, y, z = w / n, x / n, y / n, z / n

        self.qw_px4 = w
        self.qx_px4 = x
        self.qy_px4 = y
        self.qz_px4 = z

        self.vN_m = float(msg.velocity[0])
        self.vE_m = float(msg.velocity[1])
        self.vD_m = float(msg.velocity[2])

        R_nb = quat_to_R_body_to_ned(self.qw_px4, self.qx_px4, self.qy_px4, self.qz_px4)
        self.vx_body_m, self.vy_body_m, self.vz_body_m = Rt_times_v(
            R_nb, [self.vN_m, self.vE_m, self.vD_m]
        )

        self.have_odom = True

    def publish_cmd(self, bx, by, bz, yaw_rate):
        msg = Twist()
        msg.linear.x = float(bx)
        msg.linear.y = float(by)
        msg.linear.z = float(bz)
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(yaw_rate)
        self.pub.publish(msg)

    def loop(self):
        t = time.time() - self.t_start

        if not self.have_pose or not self.have_odom:
            self.publish_cmd(0.0, 0.0, 0.0, 0.0)
            return

        # Current yaw from Opti quaternion
        yaw_raw = yaw_from_quat_xyzw(self.qx_opti, self.qy_opti, self.qz_opti, self.qw_opti)
        yaw = wrap_to_pi(self.yaw_sign * yaw_raw)

        # XY errors in WORLD
        ex_xy = self.x_des - self.x_w
        ey_xy = self.y_des - self.y_w
        e_xy_norm = math.sqrt(ex_xy * ex_xy + ey_xy * ey_xy)

        # Z error in WORLD
        ez = self.z_des - self.z_w

        # Latch initial reference after warmup
        if (not self.latched) and (t >= self.warmup_s):
            self.latched = True
            self.ez_last = ez
            if e_xy_norm > 1e-9:
                self.yaw_des_last = math.atan2(-ey_xy, ex_xy)
            else:
                self.yaw_des_last = yaw

        # Warmup phase: command zeros only
        if not self.latched:
            self.publish_cmd(0.0, 0.0, 0.0, 0.0)
            self.w.writerow([
                f"{t:.3f}", "warmup",
                f"{self.x_w:.3f}", f"{self.y_w:.3f}", f"{self.z_w:.3f}",
                f"{self.x_des:.3f}", f"{self.y_des:.3f}", f"{self.z_des:.3f}",
                f"{ex_xy:.3f}", f"{ey_xy:.3f}", f"{e_xy_norm:.3f}",
                f"{ez:.3f}", "0.000",
                f"{math.degrees(yaw):.3f}", "nan", "nan",
                "0.000", "0.000", "0.000",
                "0.000",
                f"{self.vx_body_m:.3f}", f"{self.vy_body_m:.3f}", f"{self.vz_body_m:.3f}",
                "0.000", "0.000",
                "0.000",
                "0.000",
                "0.000",
                "0.000000", "0.000000", "0.000000",
            ])
            self.f.flush()
            return

        # Desired heading from CURRENT position -> desired position
        # Keep the negative sign and freeze yaw_des once inside slow-radius zone
        if e_xy_norm > self.slow_radius_xy:
            yaw_des = math.atan2(-ey_xy, ex_xy)
            self.yaw_des_last = yaw_des
        else:
            yaw_des = self.yaw_des_last

        # Yaw P controller
        yaw_err = wrap_to_pi(yaw_des - yaw)
        yaw_rate_cmd = clamp(self.kp_yaw * yaw_err, -self.yaw_rate_max, self.yaw_rate_max)

        # WORLD z PD controller
        vz_world_up_raw = self.kp_z * ez + self.kd_z * ((ez - self.ez_last) / self.dt)
        dez = (ez - self.ez_last) / self.dt
        self.ez_last = ez

        vz_world_up_cmd = clamp(vz_world_up_raw, -self.vz_world_max, self.vz_world_max)

        # Convert WORLD up command -> NED down command
        vD_des = -vz_world_up_cmd

        # BODY forward command before limiting
        if e_xy_norm > self.xy_tol:
            scale_xy = 1.0
            if self.slow_radius_xy > 1e-9:
                scale_xy = clamp(e_xy_norm / self.slow_radius_xy, 0.0, 1.0)
            vx_body_raw = self.vx_body_cmd_const * scale_xy
        else:
            vx_body_raw = 0.0

        # Speed clip to ±vx_body_max
        vel_cmd = clamp(vx_body_raw, -self.vx_body_max, self.vx_body_max)

        # Acceleration required from measured BODY x velocity
        computed_acceleration = (vel_cmd - self.vx_body_m) / self.dt

        # Acceleration limit
        if abs(computed_acceleration) > self.ax_body_max:
            vx_body_cmd = self.vx_body_prev + math.copysign(
                self.ax_body_max * self.dt,
                vel_cmd - self.vx_body_prev
            )
        else:
            vx_body_cmd = vel_cmd

        self.vx_body_prev = vx_body_cmd

        # BODY commands
        bx_cmd = vx_body_cmd
        by_cmd = 0.0

        # Solve body z so relay rotation gives desired NED vertical velocity:
        # vD = R20*bx + R21*by + R22*bz
        R_nb = quat_to_R_body_to_ned(self.qw_px4, self.qx_px4, self.qy_px4, self.qz_px4)
        R20 = R_nb[2][0]
        R21 = R_nb[2][1]
        R22 = R_nb[2][2]

        if abs(R22) > 1e-6:
            vz_body_raw = (vD_des - R20 * bx_cmd - R21 * by_cmd) / R22
        else:
            # Fallback if nearly singular
            vz_body_raw = vD_des

        bz_cmd = clamp(vz_body_raw, -self.bz_max, self.bz_max)

        # Sticky hold logic
        if not self.hold:
            if e_xy_norm <= self.xy_tol and abs(ez) <= self.z_tol:
                self.hold = True

        if self.hold:
            bx_cmd = 0.0
            by_cmd = 0.0
            self.vx_body_prev = 0.0
            # Keep z stabilized
            if abs(R22) > 1e-6:
                vz_body_raw = vD_des / R22
                bz_cmd = clamp(vz_body_raw, -self.bz_max, self.bz_max)
            else:
                vz_body_raw = vD_des
                bz_cmd = clamp(vD_des, -self.bz_max, self.bz_max)
            yaw_rate_cmd = 0.0
            phase = "hold"
        else:
            phase = "track"

        self.publish_cmd(bx_cmd, by_cmd, bz_cmd, yaw_rate_cmd)

        self.w.writerow([
            f"{t:.3f}", phase,
            f"{self.x_w:.3f}", f"{self.y_w:.3f}", f"{self.z_w:.3f}",
            f"{self.x_des:.3f}", f"{self.y_des:.3f}", f"{self.z_des:.3f}",
            f"{ex_xy:.3f}", f"{ey_xy:.3f}", f"{e_xy_norm:.3f}",
            f"{ez:.3f}", f"{dez:.3f}",
            f"{math.degrees(yaw):.3f}",
            f"{math.degrees(yaw_des):.3f}",
            f"{math.degrees(yaw_err):.3f}",
            f"{bx_cmd:.3f}", f"{by_cmd:.3f}", f"{bz_cmd:.3f}",
            f"{vx_body_raw:.3f}",
            f"{self.vx_body_m:.3f}", f"{self.vy_body_m:.3f}", f"{self.vz_body_m:.3f}",
            f"{vz_world_up_raw:.3f}", f"{vz_world_up_cmd:.3f}",
            f"{vD_des:.3f}",
            f"{vz_body_raw:.3f}",
            f"{yaw_rate_cmd:.3f}",
            f"{R20:.6f}", f"{R21:.6f}", f"{R22:.6f}",
        ])
        self.f.flush()

    def shutdown(self):
        try:
            self.publish_cmd(0.0, 0.0, 0.0, 0.0)
        except Exception:
            pass
        if self.f is not None:
            self.f.close()


def main():
    p = argparse.ArgumentParser()

    # Falcon-specific defaults
    p.add_argument("--pose-topic", default="/Drone_owl/pose")
    p.add_argument("--odom-topic", default="/owl/fmu/out/vehicle_odometry")

    p.add_argument("--x", type=float, required=True)
    p.add_argument("--y", type=float, required=True)
    p.add_argument("--z", type=float, required=True)

    p.add_argument("--vx-body", type=float, default=0.10)
    p.add_argument("--vx-body-max", type=float, default=0.30)
    p.add_argument("--ax-body-max", type=float, default=0.80)

    p.add_argument("--kp-z", type=float, default=0.6)
    p.add_argument("--kd-z", type=float, default=0.0)
    p.add_argument("--vz-world-max", type=float, default=0.30)
    p.add_argument("--bz-max", type=float, default=0.50)

    p.add_argument("--kp-yaw", type=float, default=0.6)
    p.add_argument("--yaw-rate-max", type=float, default=0.3)
    p.add_argument("--yaw-sign", type=float, default=-1.0)

    p.add_argument("--xy-tol", type=float, default=0.08)
    p.add_argument("--z-tol", type=float, default=0.05)
    p.add_argument("--slow-radius-xy", type=float, default=0.10)

    p.add_argument("--warmup", type=float, default=1.0)
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--logfile", default="/home/root/Yaw_owl.csv")

    args = p.parse_args()

    rclpy.init()
    node = YawPDController(args)
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