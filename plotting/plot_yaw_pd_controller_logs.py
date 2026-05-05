#!/usr/bin/env python3
"""
plot_yaw_pd_controller_logs.py

Plots everything possible from YawPDController_owl.py logs.

Run:
python3 plot_yaw_pd_controller_logs.py ~/Desktop/Yaw_owl.csv --show
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path):
    df = pd.read_csv(path)
    for c in df.columns:
        if c == "phase":
            df[c] = df[c].astype(str)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["t"]).copy()
    df = df.sort_values("t").reset_index(drop=True)
    df["t"] = df["t"] - df["t"].iloc[0]
    return df


def col(df, name):
    if name in df.columns:
        return df[name].to_numpy(dtype=float)
    return np.full(len(df), np.nan)


def valid(a):
    return np.isfinite(a).any()


def deriv(t, x):
    out = np.full_like(x, np.nan, dtype=float)
    good = np.isfinite(t) & np.isfinite(x)
    if good.sum() >= 2:
        idx = np.where(good)[0]
        out[idx] = np.gradient(x[idx], t[idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV file")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--prefix", default="YawPD")
    ap.add_argument("--decimate", type=int, default=1)
    args = ap.parse_args()

    df = load_csv(args.csv)
    stem = Path(args.csv).stem
    sl = slice(None, None, max(1, args.decimate))

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    def savefig(name):
        if args.outdir:
            plt.savefig(
                os.path.join(args.outdir, f"{args.prefix}_{stem}_{name}.png"),
                dpi=200,
                bbox_inches="tight",
            )

    t = col(df, "t")

    x = col(df, "x")
    y = col(df, "y")
    z = col(df, "z")
    x_des = col(df, "x_des")
    y_des = col(df, "y_des")
    z_des = col(df, "z_des")

    ex_xy = col(df, "ex_xy")
    ey_xy = col(df, "ey_xy")
    e_xy_norm = col(df, "e_xy_norm")
    ez = col(df, "ez")
    dez = col(df, "dez")

    yaw = col(df, "yaw_deg")
    yaw_des = col(df, "yaw_des_deg")
    yaw_err = col(df, "yaw_err_deg")
    yaw_rate_cmd = col(df, "yaw_rate_cmd")

    vx_cmd = col(df, "vx_body_cmd")
    vy_cmd = col(df, "vy_body_cmd")
    vz_cmd = col(df, "vz_body_cmd")

    vx_raw = col(df, "vx_body_raw")
    vy_raw = np.zeros_like(vy_cmd)
    vz_raw = col(df, "vz_body_raw")

    vx_meas = col(df, "vx_body_meas")
    vy_meas = col(df, "vy_body_meas")
    vz_meas = col(df, "vz_body_meas")

    vz_world_raw = col(df, "vz_world_up_raw")
    vz_world_cmd = col(df, "vz_world_up_cmd")
    vD_des = col(df, "vD_des")

    R20 = col(df, "R20")
    R21 = col(df, "R21")
    R22 = col(df, "R22")

    ax_raw = deriv(t, vx_raw)
    ay_raw = deriv(t, vy_raw)
    az_raw = deriv(t, vz_raw)

    ax_cmd = deriv(t, vx_cmd)
    ay_cmd = deriv(t, vy_cmd)
    az_cmd = deriv(t, vz_cmd)

    ax_meas = deriv(t, vx_meas)
    ay_meas = deriv(t, vy_meas)
    az_meas = deriv(t, vz_meas)

    v_cmd_mag = np.sqrt(vx_cmd**2 + vy_cmd**2 + vz_cmd**2)
    v_raw_mag = np.sqrt(vx_raw**2 + vy_raw**2 + vz_raw**2)
    v_meas_mag = np.sqrt(vx_meas**2 + vy_meas**2 + vz_meas**2)

    a_cmd_mag = np.sqrt(ax_cmd**2 + ay_cmd**2 + az_cmd**2)
    a_raw_mag = np.sqrt(ax_raw**2 + ay_raw**2 + az_raw**2)
    a_meas_mag = np.sqrt(ax_meas**2 + ay_meas**2 + az_meas**2)

    # 1) XY trajectory
    plt.figure()
    plt.plot(x[sl], y[sl], label="trajectory")
    plt.scatter([x[0]], [y[0]], label="start")
    plt.scatter([x[-1]], [y[-1]], label="end")
    if valid(x_des) and valid(y_des):
        plt.scatter([x_des[-1]], [y_des[-1]], marker="x", s=100, label="goal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("XY trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    savefig("xy_trajectory")

    # 2) XZ trajectory
    plt.figure()
    plt.plot(x[sl], z[sl], label="trajectory")
    if valid(x_des) and valid(z_des):
        plt.scatter([x_des[-1]], [z_des[-1]], marker="x", s=100, label="goal")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("XZ trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    savefig("xz_trajectory")

    # 3) YZ trajectory
    plt.figure()
    plt.plot(y[sl], z[sl], label="trajectory")
    if valid(y_des) and valid(z_des):
        plt.scatter([y_des[-1]], [z_des[-1]], marker="x", s=100, label="goal")
    plt.xlabel("y (m)")
    plt.ylabel("z (m)")
    plt.title("YZ trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    savefig("yz_trajectory")

    # 4) 3D trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x[sl], y[sl], z[sl], label="trajectory")
    ax.scatter([x[0]], [y[0]], [z[0]], label="start")
    ax.scatter([x[-1]], [y[-1]], [z[-1]], label="end")
    if valid(x_des) and valid(y_des) and valid(z_des):
        ax.scatter([x_des[-1]], [y_des[-1]], [z_des[-1]], marker="x", s=100, label="goal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D trajectory")
    ax.legend()
    savefig("3d_trajectory")

    # 5) Position vs desired
    plt.figure()
    plt.plot(t[sl], x[sl], label="x")
    plt.plot(t[sl], y[sl], label="y")
    plt.plot(t[sl], z[sl], label="z")
    plt.plot(t[sl], x_des[sl], "--", label="x_des")
    plt.plot(t[sl], y_des[sl], "--", label="y_des")
    plt.plot(t[sl], z_des[sl], "--", label="z_des")
    plt.xlabel("t (s)")
    plt.ylabel("position (m)")
    plt.title("Position vs desired")
    plt.grid(True)
    plt.legend()
    savefig("position_vs_desired")

    # 6) Position errors
    plt.figure()
    plt.plot(t[sl], ex_xy[sl], label="ex_xy")
    plt.plot(t[sl], ey_xy[sl], label="ey_xy")
    plt.plot(t[sl], ez[sl], label="ez")
    plt.xlabel("t (s)")
    plt.ylabel("error (m)")
    plt.title("Position errors")
    plt.grid(True)
    plt.legend()
    savefig("position_errors")

    # 7) XY distance to goal
    plt.figure()
    plt.plot(t[sl], e_xy_norm[sl], label="e_xy_norm")
    plt.xlabel("t (s)")
    plt.ylabel("distance (m)")
    plt.title("XY distance to goal")
    plt.grid(True)
    plt.legend()
    savefig("xy_distance_to_goal")

    # 8) Z error derivative
    plt.figure()
    plt.plot(t[sl], dez[sl], label="dez")
    plt.xlabel("t (s)")
    plt.ylabel("m/s")
    plt.title("Z error derivative")
    plt.grid(True)
    plt.legend()
    savefig("z_error_derivative")

    # 9) Yaw tracking
    plt.figure()
    plt.plot(t[sl], yaw[sl], label="yaw_deg")
    plt.plot(t[sl], yaw_des[sl], "--", label="yaw_des_deg")
    plt.xlabel("t (s)")
    plt.ylabel("yaw (deg)")
    plt.title("Yaw tracking")
    plt.grid(True)
    plt.legend()
    savefig("yaw_tracking")

    # 10) Yaw error
    plt.figure()
    plt.plot(t[sl], yaw_err[sl], label="yaw_err_deg")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("t (s)")
    plt.ylabel("yaw error (deg)")
    plt.title("Yaw error")
    plt.grid(True)
    plt.legend()
    savefig("yaw_error")

    # 11) Yaw rate command
    plt.figure()
    plt.plot(t[sl], yaw_rate_cmd[sl], label="yaw_rate_cmd")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("t (s)")
    plt.ylabel("rad/s")
    plt.title("Yaw rate command")
    plt.grid(True)
    plt.legend()
    savefig("yaw_rate_cmd")

    # 12-14) Body velocity components
    def plot_vel(axis, raw, cmd, meas):
        plt.figure()
        plt.plot(t[sl], raw[sl], label=f"{axis}_raw")
        plt.plot(t[sl], cmd[sl], label=f"{axis}_cmd")
        plt.plot(t[sl], meas[sl], label=f"{axis}_meas_px4")
        plt.xlabel("t (s)")
        plt.ylabel("velocity (m/s)")
        plt.title(f"{axis.upper()} body velocity: raw vs commanded vs measured")
        plt.grid(True)
        plt.legend()
        savefig(f"{axis}_body_velocity")

    plot_vel("vx", vx_raw, vx_cmd, vx_meas)
    plot_vel("vy", vy_raw, vy_cmd, vy_meas)
    plot_vel("vz", vz_raw, vz_cmd, vz_meas)

    # 15) Body velocity magnitude
    plt.figure()
    plt.plot(t[sl], v_raw_mag[sl], label="||v_body_raw||")
    plt.plot(t[sl], v_cmd_mag[sl], label="||v_body_cmd||")
    plt.plot(t[sl], v_meas_mag[sl], label="||v_body_meas_px4||")
    plt.xlabel("t (s)")
    plt.ylabel("velocity magnitude (m/s)")
    plt.title("Body velocity magnitude")
    plt.grid(True)
    plt.legend()
    savefig("body_velocity_magnitude")

    # 16) World Z / NED Z terms
    plt.figure()
    plt.plot(t[sl], vz_world_raw[sl], label="vz_world_up_raw")
    plt.plot(t[sl], vz_world_cmd[sl], label="vz_world_up_cmd")
    plt.plot(t[sl], vD_des[sl], label="vD_des")
    plt.xlabel("t (s)")
    plt.ylabel("velocity")
    plt.title("World-Z and NED-Z command terms")
    plt.grid(True)
    plt.legend()
    savefig("world_z_terms")

    # 17-19) Body acceleration components
    def plot_acc(axis, raw, cmd, meas):
        plt.figure()
        plt.plot(t[sl], raw[sl], label=f"{axis}_raw")
        plt.plot(t[sl], cmd[sl], label=f"{axis}_cmd")
        plt.plot(t[sl], meas[sl], label=f"{axis}_meas_px4")
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("t (s)")
        plt.ylabel("acceleration (m/s^2)")
        plt.title(f"{axis.upper()} body acceleration: raw vs commanded vs measured")
        plt.grid(True)
        plt.legend()
        savefig(f"{axis}_body_acceleration")

    plot_acc("ax", ax_raw, ax_cmd, ax_meas)
    plot_acc("ay", ay_raw, ay_cmd, ay_meas)
    plot_acc("az", az_raw, az_cmd, az_meas)

    # 20) Body acceleration magnitude
    plt.figure()
    plt.plot(t[sl], a_raw_mag[sl], label="||a_body_raw||")
    plt.plot(t[sl], a_cmd_mag[sl], label="||a_body_cmd||")
    plt.plot(t[sl], a_meas_mag[sl], label="||a_body_meas_px4||")
    plt.xlabel("t (s)")
    plt.ylabel("acceleration magnitude (m/s^2)")
    plt.title("Body acceleration magnitude")
    plt.grid(True)
    plt.legend()
    savefig("body_acceleration_magnitude")

    # 21) Rotation matrix row used for body-z solving
    plt.figure()
    plt.plot(t[sl], R20[sl], label="R20")
    plt.plot(t[sl], R21[sl], label="R21")
    plt.plot(t[sl], R22[sl], label="R22")
    plt.xlabel("t (s)")
    plt.ylabel("rotation matrix value")
    plt.title("Rotation row terms R20, R21, R22")
    plt.grid(True)
    plt.legend()
    savefig("rotation_terms")

    # 22) Yaw error vs forward command
    plt.figure()
    plt.plot(t[sl], yaw_err[sl], label="yaw_err_deg")
    plt.plot(t[sl], vx_cmd[sl], label="vx_body_cmd")
    plt.xlabel("t (s)")
    plt.title("Yaw error vs forward body command")
    plt.grid(True)
    plt.legend()
    savefig("yaw_error_vs_vx_cmd")

    # 23) Lateral drift insight
    plt.figure()
    plt.plot(t[sl], vy_meas[sl], label="vy_body_meas_px4")
    plt.plot(t[sl], yaw_err[sl], label="yaw_err_deg")
    plt.xlabel("t (s)")
    plt.title("Lateral measured velocity vs yaw error")
    plt.grid(True)
    plt.legend()
    savefig("vy_meas_vs_yaw_error")

    if args.show or not args.outdir:
        plt.show()


if __name__ == "__main__":
    main()