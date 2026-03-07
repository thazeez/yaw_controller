#!/usr/bin/env python3
"""
plot_yaw_controller_logs.py

Plots for YawController_OptiWorld_Clean.py CSV logs.

What it plots:
1) Measured yaw vs desired yaw
2) Yaw error vs time
3) Commanded yaw rate vs time
4) x, y, z during yaw test (to see whether the drone translated while yawing)

Run examples:

# Plot one log
python3 plot_yaw_controller_logs.py ~/Desktop/yaw_owl1.csv --show

# Plot another log
python3 plot_yaw_controller_logs.py ~/Desktop/yaw_owl2.csv --show

# Plot multiple logs
python3 plot_yaw_controller_logs.py ~/Desktop/yaw_owl1.csv ~/Desktop/yaw_owl2.csv ~/Desktop/yaw_owl3.csv ~/Desktop/yaw_owl4.csv --show

# Plot all logs at once
python3 plot_yaw_controller_logs.py ~/Desktop/yaw_owl*.csv --show

# Save figures instead of displaying
python3 plot_yaw_controller_logs.py ~/Desktop/yaw_owl*.csv --outdir ~/Desktop/yaw_plots
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def to_num(series):
    return pd.to_numeric(series, errors="coerce")


def load_csv(path):
    df = pd.read_csv(path)

    for c in df.columns:
        if c == "phase":
            df[c] = df[c].astype(str)
        else:
            df[c] = to_num(df[c])

    df = df.dropna(subset=["t"]).copy()
    return df


def sanitize_name(name):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="Yaw controller CSV files")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--prefix", default="YawController")
    ap.add_argument("--decimate", type=int, default=1)

    args = ap.parse_args()

    save = bool(args.outdir)

    if save:
        os.makedirs(args.outdir, exist_ok=True)

    dec = max(1, args.decimate)
    sl = slice(None, None, dec)

    for csv_path in args.csv:

        df = load_csv(csv_path)
        stem = sanitize_name(Path(csv_path).stem)

        t = df["t"].to_numpy()

        yaw = df.get("yaw_deg", pd.Series(np.nan, index=df.index)).to_numpy()
        yaw_des = df.get("yaw_des_deg", pd.Series(np.nan, index=df.index)).to_numpy()
        yaw_err = df.get("yaw_err_deg", pd.Series(np.nan, index=df.index)).to_numpy()
        yaw_rate_cmd = df.get("yaw_rate_cmd", pd.Series(np.nan, index=df.index)).to_numpy()

        x = df.get("x", pd.Series(np.nan, index=df.index)).to_numpy()
        y = df.get("y", pd.Series(np.nan, index=df.index)).to_numpy()
        z = df.get("z", pd.Series(np.nan, index=df.index)).to_numpy()

        def savefig(name):
            if save:
                out = os.path.join(args.outdir, f"{args.prefix}_{stem}_{name}.png")
                plt.savefig(out, dpi=200, bbox_inches="tight")

        # -------------------------------------------------
        # 1) Measured yaw vs desired yaw
        # -------------------------------------------------

        plt.figure()

        plt.plot(t[sl], yaw[sl], label="Measured yaw")

        if np.isfinite(yaw_des).any():

            yaw_des_val = yaw_des[np.isfinite(yaw_des)][0]

            # Convert -180° to 180° for visualization
            if abs(yaw_des_val + 180) < 1e-6:
                yaw_des_val = 180.0

            plt.axhline(
                yaw_des_val,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Desired yaw = {yaw_des_val:.1f}°"
            )

            plt.text(
                t[-1],
                yaw_des_val,
                f"{yaw_des_val:.1f}°",
                color="red",
                ha="right",
                va="bottom"
            )

        plt.xlabel("t (s)")
        plt.ylabel("Yaw (deg)")
        plt.title(f"Measured yaw vs desired yaw\n{Path(csv_path).name}")
        plt.legend()
        plt.grid(True)

        savefig("yaw_vs_desired")

        # -------------------------------------------------
        # 2) Yaw error vs time
        # -------------------------------------------------

        plt.figure()

        plt.plot(t[sl], yaw_err[sl], label="yaw_err_deg")
        plt.axhline(0.0, linestyle="--")

        plt.xlabel("t (s)")
        plt.ylabel("Yaw error (deg)")
        plt.title(f"Yaw error vs time\n{Path(csv_path).name}")

        plt.legend()
        plt.grid(True)

        savefig("yaw_error")

        # -------------------------------------------------
        # 3) Commanded yaw rate vs time
        # -------------------------------------------------

        plt.figure()

        plt.plot(t[sl], yaw_rate_cmd[sl], label="yaw_rate_cmd")
        plt.axhline(0.0, linestyle="--")

        plt.xlabel("t (s)")
        plt.ylabel("Yaw rate command (rad/s)")
        plt.title(f"Commanded yaw rate vs time\n{Path(csv_path).name}")

        plt.legend()
        plt.grid(True)

        savefig("yaw_rate_cmd")

        # -------------------------------------------------
        # 4) x, y, z during yaw test
        # -------------------------------------------------

        plt.figure()

        plt.plot(t[sl], x[sl], label="x")
        plt.plot(t[sl], y[sl], label="y")
        plt.plot(t[sl], z[sl], label="z")

        plt.xlabel("t (s)")
        plt.ylabel("Position (m)")
        plt.title(f"x, y, z during yaw test\n{Path(csv_path).name}")

        plt.legend()
        plt.grid(True)

        savefig("xyz_during_yaw")

        # XY drift plot

        plt.figure()

        plt.plot(x[sl], y[sl], label="XY path")

        if len(x) > 0 and np.isfinite(x[0]) and np.isfinite(y[0]):
            plt.scatter([x[0]], [y[0]], label="start")

        if len(x) > 0 and np.isfinite(x[-1]) and np.isfinite(y[-1]):
            plt.scatter([x[-1]], [y[-1]], label="end")

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"XY drift during yaw test\n{Path(csv_path).name}")

        plt.legend()
        plt.grid(True)
        plt.axis("equal")

        savefig("xy_drift")

    if args.show or not save:
        plt.show()


if __name__ == "__main__":
    main()