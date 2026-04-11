#!/usr/bin/env python3
"""
plot_yaw_face_goal_logs.py

HOW TO RUN (EXAMPLE):

python3 plot_yaw_face_goal_logs.py \
  ~/Desktop/yaw_face_goal_owl1.csv \
  --goal-x 1.3 --goal-y 1.0 \
  --show

OR SAVE PLOTS:

python3 plot_yaw_face_goal_logs.py \
  ~/Desktop/yaw_face_goal_owl1.csv \
  --goal-x 1.3 --goal-y 1.0 \
  --outdir ~/Desktop/plots

--------------------------------------------------

WHAT THIS SCRIPT DOES:

1) Plots yaw vs desired yaw
2) Plots yaw error vs time
3) Plots yaw rate command
4) Plots x,y,z vs time
5) Plots XY trajectory with:
   - goal point
   - start and end points
   - line from drone → goal
   - heading arrow (from yaw)

IMPORTANT:
- The REAL validation is numerical (atan2 geometry)
- The arrow is just visualization
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def sanitize_name(name):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def load_csv(path):
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="Face-goal CSV files")
    ap.add_argument("--goal-x", type=float, required=True)
    ap.add_argument("--goal-y", type=float, required=True)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--prefix", default="YawFaceGoal")
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

        gx = args.goal_x
        gy = args.goal_y

        # -----------------------------
        # FINAL STATE (MOST IMPORTANT)
        # -----------------------------
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(yaw)
        idx_last = np.where(valid)[0][-1]

        xf = x[idx_last]
        yf = y[idx_last]
        yawf = yaw[idx_last]

        # TRUE geometric direction to goal
        yaw_goal = np.degrees(np.arctan2(gy - yf, gx - xf))

        # error based on actual yaw
        face_err = wrap_deg(yaw_goal - yawf)

        # -----------------------------
        # ARROW (visual only)
        # -----------------------------
        # NOTE:
        # This assumes yaw_deg aligns with XY frame.
        # If arrow looks flipped, it's a convention mismatch,
        # NOT a controller issue.
        yaw_rad = np.deg2rad(yawf)
        hx = np.cos(yaw_rad)
        hy = np.sin(yaw_rad)

        def savefig(name):
            if save:
                out = os.path.join(args.outdir, f"{args.prefix}_{stem}_{name}.png")
                plt.savefig(out, dpi=200, bbox_inches="tight")

        # -----------------------------
        # 1) YAW TRACKING
        # -----------------------------
        plt.figure()
        plt.plot(t[sl], yaw[sl], label="Actual yaw")
        plt.plot(t[sl], yaw_des[sl], "--", label="Desired yaw")
        plt.xlabel("t (s)")
        plt.ylabel("Yaw (deg)")
        plt.title(f"Yaw vs Desired\n{Path(csv_path).name}")
        plt.legend()
        plt.grid(True)
        savefig("yaw_vs_desired")

        # -----------------------------
        # 2) YAW ERROR
        # -----------------------------
        plt.figure()
        plt.plot(t[sl], yaw_err[sl])
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("t (s)")
        plt.ylabel("Yaw error (deg)")
        plt.title("Yaw Error vs Time")
        plt.grid(True)
        savefig("yaw_error")

        # -----------------------------
        # 3) YAW RATE COMMAND
        # -----------------------------
        plt.figure()
        plt.plot(t[sl], yaw_rate_cmd[sl])
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("t (s)")
        plt.ylabel("Yaw rate (rad/s)")
        plt.title("Yaw Rate Command")
        plt.grid(True)
        savefig("yaw_rate")

        # -----------------------------
        # 4) POSITION
        # -----------------------------
        plt.figure()
        plt.plot(t[sl], x[sl], label="x")
        plt.plot(t[sl], y[sl], label="y")
        plt.plot(t[sl], z[sl], label="z")
        plt.legend()
        plt.xlabel("t (s)")
        plt.ylabel("Position (m)")
        plt.title("Position vs Time")
        plt.grid(True)
        savefig("xyz")

        # -----------------------------
        # 5) FACE-GOAL CHECK (KEY FIGURE)
        # -----------------------------
        plt.figure()

        plt.plot(x[sl], y[sl], label="trajectory")
        plt.scatter(x[0], y[0], label="start")
        plt.scatter(xf, yf, label="end")
        plt.scatter(gx, gy, marker="x", s=100, label="goal")

        # line to goal
        plt.plot([xf, gx], [yf, gy], "--", label="direction to goal")

        # heading arrow
        plt.arrow(
            xf, yf,
            0.3 * hx, 0.3 * hy,
            head_width=0.05,
            length_includes_head=True
        )

        plt.axis("equal")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        plt.title(
            f"Face Goal Check\n{Path(csv_path).name}\n"
            f"final yaw = {yawf:.2f}° | "
            f"goal yaw = {yaw_goal:.2f}° | "
            f"error = {face_err:.2f}°"
        )

        plt.legend()
        plt.grid(True)
        savefig("face_goal")

    if args.show or not save:
        plt.show()


if __name__ == "__main__":
    main()