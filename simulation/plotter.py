"""
Project INTERCEPTOR — Matplotlib Trajectory Plotter
===================================================

Generates publication-quality engagement plots:
  1. 3D trajectory — interceptor vs target flight paths
  2. Telemetry dashboard — distance, ZEM, G-load, closing velocity over time

Saves output to output/ directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving to file
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


# ── Style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "axes.grid": True,
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "font.family": "monospace",
    "font.size": 10,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.fontsize": 9,
})

INTERCEPTOR_COLOR = "#58a6ff"
TARGET_COLOR = "#f85149"
ZEM_COLOR = "#d29922"
GLOAD_COLOR = "#3fb950"
CLOSING_COLOR = "#bc8cff"
ACCENT_CYAN = "#39d353"


def collect_engagement_data(env, max_steps: int = 5000):
    """
    Run one engagement episode and collect full trajectory data.
    
    Returns:
        dict with arrays: interceptor_pos, target_pos, distances,
        zem_norms, g_loads, closing_vels, times, hit, min_dist
    """
    obs, _ = env.reset()
    
    data = {
        "interceptor_pos": [],
        "target_pos": [],
        "distances": [],
        "zem_norms": [],
        "g_loads": [],
        "closing_vels": [],
        "times": [],
        "accel_y": [],
        "accel_z": [],
    }
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        raw = env.get_raw_state()
        step += 1
        done = terminated or truncated
        
        data["interceptor_pos"].append(raw["interceptor_pos"])
        data["target_pos"].append(raw["target_pos"])
        data["distances"].append(raw["distance"])
        data["zem_norms"].append(raw["zem_norm"])
        data["g_loads"].append(np.linalg.norm(action) / 9.81)
        data["closing_vels"].append(raw["closing_velocity"])
        data["times"].append(raw["time_seconds"])
        data["accel_y"].append(action[0])
        data["accel_z"].append(action[1])
    
    stats = info.get("episode_stats", {})
    data["hit"] = stats.get("hit", False)
    data["min_distance"] = stats.get("min_distance", float("inf"))
    data["total_steps"] = step
    
    # Convert to numpy arrays
    for key in ["interceptor_pos", "target_pos", "distances", "zem_norms",
                "g_loads", "closing_vels", "times", "accel_y", "accel_z"]:
        data[key] = np.array(data[key])
    
    return data


def plot_3d_trajectory(ax, data: dict):
    """Plot the 3D flight paths of interceptor and target."""
    
    ipos = data["interceptor_pos"]
    tpos = data["target_pos"]
    
    # Interceptor trajectory
    ax.plot(
        ipos[:, 0], ipos[:, 1], ipos[:, 2],
        color=INTERCEPTOR_COLOR, linewidth=2, label="Interceptor",
        alpha=0.9
    )
    # Target trajectory
    ax.plot(
        tpos[:, 0], tpos[:, 1], tpos[:, 2],
        color=TARGET_COLOR, linewidth=2, label="Target",
        alpha=0.9, linestyle="--"
    )
    
    # Start markers
    ax.scatter(*ipos[0], color=INTERCEPTOR_COLOR, s=80, marker="^",
               edgecolors="white", linewidths=0.5, zorder=5)
    ax.scatter(*tpos[0], color=TARGET_COLOR, s=80, marker="D",
               edgecolors="white", linewidths=0.5, zorder=5)
    
    # End markers (larger)
    ax.scatter(*ipos[-1], color=INTERCEPTOR_COLOR, s=120, marker="^",
               edgecolors="white", linewidths=1, zorder=6, label="Final (Interceptor)")
    ax.scatter(*tpos[-1], color=TARGET_COLOR, s=120, marker="D",
               edgecolors="white", linewidths=1, zorder=6, label="Final (Target)")
    
    # Closest approach line
    min_idx = np.argmin(data["distances"])
    ax.plot(
        [ipos[min_idx, 0], tpos[min_idx, 0]],
        [ipos[min_idx, 1], tpos[min_idx, 1]],
        [ipos[min_idx, 2], tpos[min_idx, 2]],
        color=ZEM_COLOR, linewidth=1.5, linestyle=":", alpha=0.8,
        label=f"Closest approach: {data['min_distance']:.1f}m"
    )
    
    ax.set_xlabel("X  (downrange) [m]", labelpad=8)
    ax.set_ylabel("Y  (crossrange) [m]", labelpad=8)
    ax.set_zlabel("Z  (altitude) [m]", labelpad=8)
    ax.set_title("3D ENGAGEMENT TRAJECTORY", fontsize=13, fontweight="bold",
                 color="#58a6ff", pad=15)
    
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#30363d")
    ax.yaxis.pane.set_edgecolor("#30363d")
    ax.zaxis.pane.set_edgecolor("#30363d")
    
    ax.view_init(elev=25, azim=-60)


def plot_telemetry_dashboard(axes, data: dict):
    """Plot the 4-panel telemetry dashboard."""
    t = data["times"]
    
    # ── Panel 1: Distance ──
    ax1 = axes[0]
    ax1.fill_between(t, data["distances"], alpha=0.15, color=INTERCEPTOR_COLOR)
    ax1.plot(t, data["distances"], color=INTERCEPTOR_COLOR, linewidth=1.5)
    ax1.axhline(y=5.0, color=ACCENT_CYAN, linewidth=1, linestyle="--", alpha=0.6, label="Kill radius (5m)")
    
    min_idx = np.argmin(data["distances"])
    ax1.annotate(
        f"  MIN: {data['min_distance']:.1f}m",
        xy=(t[min_idx], data["distances"][min_idx]),
        fontsize=9, fontweight="bold", color=ZEM_COLOR,
        arrowprops=dict(arrowstyle="->", color=ZEM_COLOR, lw=1.2),
        xytext=(t[min_idx] - 0.5, data["distances"][min_idx] + 1500),
    )
    
    ax1.set_ylabel("Distance [m]")
    ax1.set_title("RANGE TO TARGET", fontsize=10, fontweight="bold", color=INTERCEPTOR_COLOR)
    ax1.legend(loc="upper right", fontsize=8)
    
    # ── Panel 2: ZEM ──
    ax2 = axes[1]
    ax2.fill_between(t, data["zem_norms"], alpha=0.15, color=ZEM_COLOR)
    ax2.plot(t, data["zem_norms"], color=ZEM_COLOR, linewidth=1.5)
    ax2.set_ylabel("ZEM [m]")
    ax2.set_title("ZERO-EFFORT-MISS", fontsize=10, fontweight="bold", color=ZEM_COLOR)
    
    # ── Panel 3: G-Load ──
    ax3 = axes[2]
    ax3.fill_between(t, data["g_loads"], alpha=0.15, color=GLOAD_COLOR)
    ax3.plot(t, data["g_loads"], color=GLOAD_COLOR, linewidth=1, alpha=0.7)
    ax3.axhline(y=30, color="#f85149", linewidth=1, linestyle="--", alpha=0.5, label="Max G (30G)")
    ax3.set_ylabel("G-Load [G]")
    ax3.set_title("COMMANDED ACCELERATION", fontsize=10, fontweight="bold", color=GLOAD_COLOR)
    ax3.legend(loc="upper right", fontsize=8)
    
    # ── Panel 4: Closing Velocity ──
    ax4 = axes[3]
    ax4.fill_between(t, data["closing_vels"], alpha=0.15, color=CLOSING_COLOR)
    ax4.plot(t, data["closing_vels"], color=CLOSING_COLOR, linewidth=1.5)
    ax4.set_ylabel("V_c [m/s]")
    ax4.set_xlabel("Time [s]")
    ax4.set_title("CLOSING VELOCITY", fontsize=10, fontweight="bold", color=CLOSING_COLOR)


def generate_engagement_plot(env, output_dir: str = "output"):
    """
    Run one engagement, then generate and save a full plot.
    
    Returns:
        Path to the saved image.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("  Collecting engagement data...")
    data = collect_engagement_data(env)
    
    result_str = "HIT" if data["hit"] else "MISS"
    print(f"  Engagement: {result_str} | Min dist: {data['min_distance']:.2f}m | Steps: {data['total_steps']}")
    
    # ── Create figure ──
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "PROJECT INTERCEPTOR — ENGAGEMENT ANALYSIS",
        fontsize=16, fontweight="bold", color="#58a6ff",
        y=0.98
    )
    
    # Subtitle with stats
    fig.text(
        0.5, 0.955,
        f"Result: {result_str}  │  Min Distance: {data['min_distance']:.1f}m  │  "
        f"Duration: {data['times'][-1]:.2f}s  │  Steps: {data['total_steps']}",
        ha="center", fontsize=10, color="#8b949e"
    )
    
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3,
                  left=0.06, right=0.97, top=0.92, bottom=0.05)
    
    # 3D trajectory (top row, spanning both columns)
    ax_3d = fig.add_subplot(gs[0, :], projection="3d")
    plot_3d_trajectory(ax_3d, data)
    
    # Telemetry panels (bottom two rows)
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_zem = fig.add_subplot(gs[1, 1])
    ax_gload = fig.add_subplot(gs[2, 0])
    ax_closv = fig.add_subplot(gs[2, 1])
    
    plot_telemetry_dashboard(
        [ax_dist, ax_zem, ax_gload, ax_closv],
        data
    )
    
    # Save
    output_path = os.path.join(output_dir, "engagement_plot.png")
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    
    print(f"  ✅ Plot saved to: {output_path}")
    return output_path


def generate_multi_engagement_plot(env, n_episodes: int = 5, output_dir: str = "output"):
    """
    Run multiple engagements and generate a comparison dashboard.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_data = []
    print(f"  Running {n_episodes} engagements...")
    
    for i in range(n_episodes):
        data = collect_engagement_data(env)
        all_data.append(data)
        status = "HIT ✓" if data["hit"] else "MISS ✗"
        print(f"    [{i+1}/{n_episodes}] {status} | Min: {data['min_distance']:.1f}m | Steps: {data['total_steps']}")
    
    # ── Summary plot ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"PROJECT INTERCEPTOR — {n_episodes}-ENGAGEMENT SUMMARY",
        fontsize=14, fontweight="bold", color="#58a6ff", y=0.98
    )
    
    hits = sum(1 for d in all_data if d["hit"])
    fig.text(
        0.5, 0.955,
        f"Hit Rate: {hits}/{n_episodes} ({hits/n_episodes*100:.0f}%)  │  "
        f"Avg Min Distance: {np.mean([d['min_distance'] for d in all_data]):.1f}m",
        ha="center", fontsize=10, color="#8b949e"
    )
    
    # Panel 1: Distance overlay
    ax1 = axes[0, 0]
    for i, d in enumerate(all_data):
        color = ACCENT_CYAN if d["hit"] else TARGET_COLOR
        alpha = 0.8 if d["hit"] else 0.4
        ax1.plot(d["times"], d["distances"], color=color, alpha=alpha,
                 linewidth=1, label=f"Eng {i+1}" if i < 3 else None)
    ax1.set_ylabel("Distance [m]")
    ax1.set_title("RANGE (all engagements)", fontsize=10, fontweight="bold", color=INTERCEPTOR_COLOR)
    ax1.legend(fontsize=7, ncol=2)
    
    # Panel 2: ZEM overlay
    ax2 = axes[0, 1]
    for i, d in enumerate(all_data):
        color = ACCENT_CYAN if d["hit"] else TARGET_COLOR
        ax2.plot(d["times"], d["zem_norms"], color=color, alpha=0.5, linewidth=1)
    ax2.set_ylabel("ZEM [m]")
    ax2.set_title("ZERO-EFFORT-MISS (all engagements)", fontsize=10, fontweight="bold", color=ZEM_COLOR)
    
    # Panel 3: Min distance bar chart
    ax3 = axes[1, 0]
    labels = [f"ENG-{i+1}" for i in range(n_episodes)]
    min_dists = [d["min_distance"] for d in all_data]
    colors = [ACCENT_CYAN if d["hit"] else TARGET_COLOR for d in all_data]
    bars = ax3.bar(labels, min_dists, color=colors, alpha=0.8, edgecolor="#30363d")
    ax3.axhline(y=5.0, color=ACCENT_CYAN, linewidth=1, linestyle="--", alpha=0.6, label="Kill radius")
    ax3.set_ylabel("Min Distance [m]")
    ax3.set_title("CLOSEST APPROACH", fontsize=10, fontweight="bold", color=GLOAD_COLOR)
    ax3.legend(fontsize=8)
    
    # Add value labels on bars
    for bar, val in zip(bars, min_dists):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f"{val:.0f}m", ha="center", fontsize=8, color="#c9d1d9")
    
    # Panel 4: G-load distribution
    ax4 = axes[1, 1]
    for i, d in enumerate(all_data):
        ax4.hist(d["g_loads"], bins=30, alpha=0.3, color=GLOAD_COLOR, edgecolor="none")
    ax4.axvline(x=30, color="#f85149", linewidth=1.5, linestyle="--", alpha=0.7, label="Max G")
    ax4.set_xlabel("G-Load [G]")
    ax4.set_ylabel("Frequency")
    ax4.set_title("G-LOAD DISTRIBUTION", fontsize=10, fontweight="bold", color=CLOSING_COLOR)
    ax4.legend(fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    output_path = os.path.join(output_dir, "multi_engagement.png")
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    
    print(f"\n  ✅ Multi-engagement plot saved to: {output_path}")
    return output_path
