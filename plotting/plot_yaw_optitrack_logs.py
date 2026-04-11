#!/usr/bin/env python3
"""
plot_yaw_optitrack_logs.py

For CSVs with columns:
t, phase, x, y, z, yaw_deg, yaw_des_deg, yaw_err_deg, yaw_rate_cmd
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path):
    return pd.read_csv(path)


def sanitize_name(name):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="OptiTrack yaw log CSV files")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--prefix", default="YawOptiTrack")
    ap.add_argument("--decimate", type=int, default=1)
    args = ap.parse_args()

    save = bool(args.outdir)
    if save:
        os.makedirs(args.outdir, exist_ok=True)

    sl = slice(None, None, max(1, args.decimate))

    for csv_path in args.csv:
        df = load_csv(csv_path)
        stem = sanitize_name(Path(csv_path).stem)

        t = pd.to_numeric(df["t"], errors="coerce").to_numpy()
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy()
        y = pd.to_numeric(df["y"], errors="coerce").to_numpy()
        z = pd.to_numeric(df["z"], errors="coerce").to_numpy()
        yaw = pd.to_numeric(df["yaw_deg"], errors="coerce").to_numpy()
        yaw_des = pd.to_numeric(df["yaw_des_deg"], errors="coerce").to_numpy()
        yaw_err = pd.to_numeric(df["yaw_err_deg"], errors="coerce").to_numpy()
        yaw_rate_cmd = pd.to_numeric(df["yaw_rate_cmd"], errors="coerce").to_numpy()

        def savefig(name):
            if save:
                out = os.path.join(args.outdir, f"{args.prefix}_{stem}_{name}.png")
                plt.savefig(out, dpi=200, bbox_inches="tight")

        # 1) Commanded yaw vs OptiTrack yaw
        plt.figure()
        plt.plot(t[sl], yaw[sl], label="OptiTrack yaw")
        plt.plot(t[sl], yaw_des[sl], "--", linewidth=2, label="Commanded yaw")
        plt.xlabel("t (s)")
        plt.ylabel("Yaw (deg)")
        plt.title(f"Commanded yaw vs OptiTrack yaw\n{Path(csv_path).name}")
        plt.legend()
        plt.grid(True)
        savefig("yaw_vs_command")

        # 2) Yaw error
        plt.figure()
        plt.plot(t[sl], yaw_err[sl], label="Yaw error")
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("t (s)")
        plt.ylabel("Yaw error (deg)")
        plt.title(f"Yaw error vs time\n{Path(csv_path).name}")
        plt.legend()
        plt.grid(True)
        savefig("yaw_error")

        # 3) Commanded yaw rate
        plt.figure()
        plt.plot(t[sl], yaw_rate_cmd[sl], label="Commanded yaw rate")
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("t (s)")
        plt.ylabel("Yaw rate command")
        plt.title(f"Commanded yaw rate vs time\n{Path(csv_path).name}")
        plt.legend()
        plt.grid(True)
        savefig("yaw_rate_cmd")

        # 4) x, y, z vs time
        plt.figure()
        plt.plot(t[sl], x[sl], label="x")
        plt.plot(t[sl], y[sl], label="y")
        plt.plot(t[sl], z[sl], label="z")
        plt.xlabel("t (s)")
        plt.ylabel("Position (m)")
        plt.title(f"x, y, z vs time\n{Path(csv_path).name}")
        plt.legend()
        plt.grid(True)
        savefig("xyz_vs_time")

        # 5) XY trajectory
        plt.figure()
        plt.plot(x[sl], y[sl], label="XY trajectory")
        plt.scatter([x[0]], [y[0]], label="start")
        plt.scatter([x[-1]], [y[-1]], label="end")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"XY trajectory\n{Path(csv_path).name}")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        savefig("xy_trajectory")

        # 6) XZ trajectory
        plt.figure()
        plt.plot(x[sl], z[sl], label="XZ trajectory")
        plt.scatter([x[0]], [z[0]], label="start")
        plt.scatter([x[-1]], [z[-1]], label="end")
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        plt.title(f"XZ trajectory\n{Path(csv_path).name}")
        plt.legend()
        plt.grid(True)
        savefig("xz_trajectory")

        # 7) YZ trajectory
        plt.figure()
        plt.plot(y[sl], z[sl], label="YZ trajectory")
        plt.scatter([y[0]], [z[0]], label="start")
        plt.scatter([y[-1]], [z[-1]], label="end")
        plt.xlabel("y (m)")
        plt.ylabel("z (m)")
        plt.title(f"YZ trajectory\n{Path(csv_path).name}")
        plt.legend()
        plt.grid(True)
        savefig("yz_trajectory")

    if args.show or not save:
        plt.show()


if __name__ == "__main__":
    main()