#!/usr/bin/env python3
#how to run: python3 offboard_vel_relay_body3_fullquat_withyaw.py --cmd-timeout 0.3 --offboard-prestream 0.8 --yawrate-max 1.0


"""
OFFBOARD VELOCITY RELAY (BODY COMMANDS IN -> PX4 LOCAL OUT)  [FULL QUATERNION ROTATION]
+ YAW RATE FORWARDING (cmd_vel.angular.z -> TrajectorySetpoint.yawspeed)

- NO TAKEOFF, NO ARM
- Start while already hovering in Position mode (armed + Position/Altitude hold)
- Prestreams zero velocity, then switches to OFFBOARD

Subscribes to /cmd_vel (geometry_msgs/Twist) interpreted as PX4 BODY frame (FRD):
  linear:
    x = forward (m/s), y = right (m/s), z = down (m/s)
  angular:
    z = yaw rate command (rad/s)   ✅ NEW (forwarded)

- Rotates BODY(FRD) linear -> PX4 local NED using FULL attitude quaternion from VehicleOdometry
- Publishes PX4 TrajectorySetpoint.velocity AND TrajectorySetpoint.yawspeed continuously
- If /cmd_vel is stale, commands zero velocity AND zero yawspeed

Notes:
- PX4 VehicleOdometry.q is WXYZ in your system (verified): q = [w,x,y,z]
- PX4 topics use BEST_EFFORT QoS.
"""

import math
import csv
import argparse
import os
import uuid
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry
from geometry_msgs.msg import Twist


def qos_px4(depth=10):
    q = QoSProfile(depth=depth)
    q.history = HistoryPolicy.KEEP_LAST
    q.reliability = ReliabilityPolicy.BEST_EFFORT
    q.durability = DurabilityPolicy.VOLATILE
    return q


def qos_cmd(depth=10):
    q = QoSProfile(depth=depth)
    q.history = HistoryPolicy.KEEP_LAST
    q.reliability = ReliabilityPolicy.RELIABLE
    q.durability = DurabilityPolicy.VOLATILE
    return q


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def quat_to_R_body_to_ned(w, x, y, z):
    """
    Rotation matrix R such that:
        v_ned = R * v_body
    where:
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


def R_times_v(R, v):
    return [
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    ]


class OffboardVelRelayBodyFullQuat(Node):
    def __init__(
        self,
        cmd_topic="/cmd_vel",
        hz=50.0,
        offboard_prestream=0.8,
        cmd_timeout=0.3,
        vmax_xy=0.8,
        vmax_z=0.6,
        yawrate_max=1.0,          # ✅ NEW
        require_first_cmd=True,
        log_path="",
    ):
        super().__init__("offboard_vel_relay_body_fullquat_withyaw")

        self.dt = 1.0 / float(hz)
        self.t0 = self.get_clock().now()
        self.offboard_prestream = float(max(0.1, offboard_prestream))

        self.cmd_timeout = float(max(0.0, cmd_timeout))
        self.vmax_xy = float(abs(vmax_xy))
        self.vmax_z = float(abs(vmax_z))
        self.yawrate_max = float(abs(yawrate_max))   # ✅ NEW
        self.require_first_cmd = bool(require_first_cmd)

        self.have_odom = False

        # current attitude quaternion (WXYZ) from PX4 odom
        self.qw, self.qx, self.qy, self.qz = 1.0, 0.0, 0.0, 0.0

        # latest /cmd_vel in body FRD
        self.cmd_bx = 0.0
        self.cmd_by = 0.0
        self.cmd_bz = 0.0
        self.cmd_r  = 0.0          # ✅ NEW: yaw rate command (rad/s), from angular.z
        self.last_cmd_t = None
        self.seen_first_cmd = False

        self.phase = "init"

        self.pub_cm = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos_px4())
        self.pub_sp = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_px4())
        self.pub_cmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos_px4())

        self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self._odom_cb, qos_px4())
        self.create_subscription(Twist, cmd_topic, self._cmd_cb, qos_cmd())

        run_id = uuid.uuid4().hex[:8]
        if not log_path:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            os.makedirs("/home/root/logs", exist_ok=True)
            log_path = f"/home/root/logs/offboard_body_relay_fullquat_withyaw_{ts}_{run_id}.csv"

        self.log_f = open(log_path, "w", newline="")
        self.log_w = csv.writer(self.log_f)
        self.log_w.writerow([
            "t_sec", "phase",
            "bx", "by", "bz", "r_cmd",
            "vx_sp", "vy_sp", "vz_sp", "yawspeed_sp",
            "x", "y", "z",
            "vx_m", "vy_m", "vz_m",
            "stale",
            "qw", "qx", "qy", "qz",
        ])

        self.timer = self.create_timer(self.dt, self._loop)

        self.get_logger().info("OFFBOARD BODY-VELOCITY RELAY started (FULL QUAT + YAW, NO ARM/NO TAKEOFF)")
        self.get_logger().info(f"Listening on {cmd_topic} as BODY cmd: x=fwd, y=right, z=down, yawrate=angular.z")
        self.get_logger().info(f"hz={hz:.1f}, prestream={self.offboard_prestream:.2f}s, cmd_timeout={self.cmd_timeout:.2f}s")
        self.get_logger().info(f"limits: vmax_xy={self.vmax_xy:.2f}, vmax_z={self.vmax_z:.2f}, yawrate_max={self.yawrate_max:.2f}")
        self.get_logger().info(f"Logging to: {log_path}")

    def _now_s(self):
        return (self.get_clock().now() - self.t0).nanoseconds * 1e-9

    def _now_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def _odom_cb(self, m: VehicleOdometry):
        w, x, y, z = float(m.q[0]), float(m.q[1]), float(m.q[2]), float(m.q[3])
        n = math.sqrt(w*w + x*x + y*y + z*z)
        if n > 1e-9:
            w, x, y, z = w/n, x/n, y/n, z/n
        self.qw, self.qx, self.qy, self.qz = w, x, y, z
        self.have_odom = True

        # telemetry for logging
        self.x = float(m.position[0])
        self.y = float(m.position[1])
        self.z = float(m.position[2])
        self.vx_m = float(m.velocity[0])
        self.vy_m = float(m.velocity[1])
        self.vz_m = float(m.velocity[2])

    def _cmd_cb(self, msg: Twist):
        self.cmd_bx = float(msg.linear.x)
        self.cmd_by = float(msg.linear.y)
        self.cmd_bz = float(msg.linear.z)
        self.cmd_r  = float(msg.angular.z)   # ✅ NEW
        self.last_cmd_t = self._now_s()
        self.seen_first_cmd = True

    def _loop(self):
        t = self._now_s()

        # Always publish OffboardControlMode (velocity control)
        cm = OffboardControlMode()
        cm.timestamp = self._now_us()
        cm.position = False
        cm.velocity = True
        cm.acceleration = False
        cm.attitude = False
        cm.body_rate = False
        self.pub_cm.publish(cm)

        stale = True
        if self.last_cmd_t is not None:
            stale = (t - self.last_cmd_t) > self.cmd_timeout

        if stale:
            bx, by, bz, r_cmd = 0.0, 0.0, 0.0, 0.0
        else:
            bx, by, bz = self.cmd_bx, self.cmd_by, self.cmd_bz
            r_cmd = clamp(self.cmd_r, -self.yawrate_max, self.yawrate_max)

        # BODY(FRD) linear -> NED using quaternion
        R_nb = quat_to_R_body_to_ned(self.qw, self.qx, self.qy, self.qz)
        vN, vE, vD = R_times_v(R_nb, [bx, by, bz])

        # Clamp linear in NED
        vx = clamp(vN, -self.vmax_xy, self.vmax_xy)
        vy = clamp(vE, -self.vmax_xy, self.vmax_xy)
        vz = clamp(vD, -self.vmax_z, self.vmax_z)

        # Publish TrajectorySetpoint (velocity + yawspeed)
        nan = float("nan")
        sp = TrajectorySetpoint()
        sp.timestamp = self._now_us()
        sp.position = [nan, nan, nan]
        sp.acceleration = [nan, nan, nan]
        sp.jerk = [nan, nan, nan]
        sp.yaw = nan
        sp.yawspeed = float(r_cmd)     # ✅ NEW: finite yaw rate goes to PX4
        sp.velocity = [vx, vy, vz]
        self.pub_sp.publish(sp)

        # Log
        self.log_w.writerow([
            f"{t:.6f}", self.phase,
            bx, by, bz, r_cmd,
            vx, vy, vz, r_cmd,
            getattr(self, "x", nan), getattr(self, "y", nan), getattr(self, "z", nan),
            getattr(self, "vx_m", nan), getattr(self, "vy_m", nan), getattr(self, "vz_m", nan),
            int(stale),
            self.qw, self.qx, self.qy, self.qz,
        ])
        self.log_f.flush()

        # Switch to OFFBOARD once ready
        if self.phase == "init" and self.have_odom and (self.seen_first_cmd or not self.require_first_cmd) and t >= self.offboard_prestream:
            cmd = VehicleCommand()
            cmd.timestamp = self._now_us()
            cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
            cmd.param1 = 1.0
            cmd.param2 = 6.0  # PX4 offboard mode
            cmd.target_system = 1
            cmd.target_component = 1
            cmd.from_external = True
            self.pub_cmd.publish(cmd)
            self.phase = "offboard"
            self.get_logger().info("Phase: OFFBOARD (relay active, FULL QUAT + YAW)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd-topic", type=str, default="/cmd_vel")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--offboard-prestream", type=float, default=0.8)
    parser.add_argument("--cmd-timeout", type=float, default=0.3)
    parser.add_argument("--vmax-xy", type=float, default=0.8)
    parser.add_argument("--vmax-z", type=float, default=0.6)
    parser.add_argument("--yawrate-max", type=float, default=1.0)   # ✅ NEW
    parser.add_argument("--require-first-cmd", action="store_true", default=True)
    parser.add_argument("--log-path", type=str, default="")
    args = parser.parse_args()

    rclpy.init()
    node = OffboardVelRelayBodyFullQuat(
        cmd_topic=args.cmd_topic,
        hz=args.hz,
        offboard_prestream=args.offboard_prestream,
        cmd_timeout=args.cmd_timeout,
        vmax_xy=args.vmax_xy,
        vmax_z=args.vmax_z,
        yawrate_max=args.yawrate_max,
        require_first_cmd=args.require_first_cmd,
        log_path=args.log_path,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.log_f.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
