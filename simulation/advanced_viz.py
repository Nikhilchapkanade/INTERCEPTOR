"""
Project INTERCEPTOR — Advanced Visualizations
=============================================

Three high-impact visualizations for README and portfolio:
  1. Animated engagement GIF  (FuncAnimation)
  2. Top-down engagement heatmap (infrared aesthetic)
  3. Before/After training comparison
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch
from pathlib import Path

from simulation.plotter import (
    collect_engagement_data,
    INTERCEPTOR_COLOR, TARGET_COLOR, ZEM_COLOR, GLOAD_COLOR,
    CLOSING_COLOR, ACCENT_CYAN,
)


# ── Shared dark style ──
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
})


# ═══════════════════════════════════════════════════════════════
#  1. ANIMATED ENGAGEMENT GIF
# ═══════════════════════════════════════════════════════════════

def generate_animated_engagement(env, output_dir: str = "output", fps: int = 30):
    """
    Create an animated GIF showing the interceptor chasing the target
    in real-time with a telemetry overlay.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("    Collecting trajectory data...")
    data = collect_engagement_data(env)
    
    ipos = data["interceptor_pos"]
    tpos = data["target_pos"]
    n_frames = len(ipos)
    
    # Subsample frames for reasonable GIF size (target ~200 frames)
    step = max(1, n_frames // 200)
    frame_indices = list(range(0, n_frames, step))
    if frame_indices[-1] != n_frames - 1:
        frame_indices.append(n_frames - 1)
    
    print(f"    Rendering {len(frame_indices)} frames...")
    
    # ── Figure setup ──
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2, 1],
                  left=0.06, right=0.97, top=0.90, bottom=0.08, wspace=0.05)
    
    # Left panel: 2D top-down trajectory (X vs Y)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_traj.set_facecolor("#0a0e14")
    ax_traj.set_xlabel("X (downrange) [m]")
    ax_traj.set_ylabel("Y (crossrange) [m]")
    ax_traj.set_title("REAL-TIME ENGAGEMENT", fontsize=13, fontweight="bold",
                       color="#58a6ff", pad=10)
    
    # Set axis limits based on full trajectory
    all_x = np.concatenate([ipos[:, 0], tpos[:, 0]])
    all_y = np.concatenate([ipos[:, 1], tpos[:, 1]])
    pad_x = (all_x.max() - all_x.min()) * 0.1 + 100
    pad_y = (all_y.max() - all_y.min()) * 0.15 + 200
    ax_traj.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax_traj.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)
    ax_traj.set_aspect("auto")
    
    # Grid styling
    ax_traj.grid(True, alpha=0.15, color="#30363d")
    
    # Persistent artists
    int_trail, = ax_traj.plot([], [], color=INTERCEPTOR_COLOR, linewidth=1.5, alpha=0.6)
    tgt_trail, = ax_traj.plot([], [], color=TARGET_COLOR, linewidth=1.5, alpha=0.6, linestyle="--")
    int_dot, = ax_traj.plot([], [], "^", color=INTERCEPTOR_COLOR, markersize=12,
                             markeredgecolor="white", markeredgewidth=0.8)
    tgt_dot, = ax_traj.plot([], [], "D", color=TARGET_COLOR, markersize=10,
                             markeredgecolor="white", markeredgewidth=0.8)
    los_line, = ax_traj.plot([], [], color=ZEM_COLOR, linewidth=0.8, alpha=0.5, linestyle=":")
    
    # Start markers (static)
    ax_traj.plot(ipos[0, 0], ipos[0, 1], "^", color=INTERCEPTOR_COLOR, markersize=8, alpha=0.3)
    ax_traj.plot(tpos[0, 0], tpos[0, 1], "D", color=TARGET_COLOR, markersize=8, alpha=0.3)
    
    # Right panel: telemetry text
    ax_tel = fig.add_subplot(gs[0, 1])
    ax_tel.set_facecolor("#0a0e14")
    ax_tel.axis("off")
    ax_tel.set_xlim(0, 1)
    ax_tel.set_ylim(0, 1)
    
    # Static telemetry labels
    label_props = dict(fontsize=9, fontfamily="monospace", color="#8b949e",
                       transform=ax_tel.transAxes, ha="left", va="top")
    value_props = dict(fontsize=12, fontfamily="monospace", fontweight="bold",
                       transform=ax_tel.transAxes, ha="left", va="top")
    
    ax_tel.text(0.05, 0.95, "TELEMETRY", fontsize=14, fontweight="bold",
                color="#58a6ff", transform=ax_tel.transAxes)
    ax_tel.text(0.05, 0.88, "─" * 22, color="#30363d", fontsize=9,
                transform=ax_tel.transAxes, fontfamily="monospace")
    
    labels = ["DISTANCE", "ZEM", "CLOSING VEL", "TIME-TO-GO", "G-LOAD", "PHASE", "SIM TIME"]
    y_positions = [0.82, 0.72, 0.62, 0.52, 0.42, 0.30, 0.20]
    
    for label, y in zip(labels, y_positions):
        ax_tel.text(0.05, y, label, **label_props)
    
    # Dynamic value texts (will be updated each frame)
    value_texts = []
    for y in y_positions:
        txt = ax_tel.text(0.05, y - 0.04, "", **value_props)
        value_texts.append(txt)
    
    # Result text (shown at end)
    result_text = ax_tel.text(0.5, 0.08, "", fontsize=14, fontweight="bold",
                               transform=ax_tel.transAxes, ha="center", va="center")
    
    # Title
    fig.suptitle("PROJECT INTERCEPTOR — ENGAGEMENT REPLAY",
                 fontsize=14, fontweight="bold", color="#58a6ff", y=0.97)
    
    def update(frame_num):
        idx = frame_indices[frame_num]
        
        # Update trails
        int_trail.set_data(ipos[:idx+1, 0], ipos[:idx+1, 1])
        tgt_trail.set_data(tpos[:idx+1, 0], tpos[:idx+1, 1])
        
        # Update current positions
        int_dot.set_data([ipos[idx, 0]], [ipos[idx, 1]])
        tgt_dot.set_data([tpos[idx, 0]], [tpos[idx, 1]])
        
        # LOS line
        los_line.set_data([ipos[idx, 0], tpos[idx, 0]], [ipos[idx, 1], tpos[idx, 1]])
        
        # Update telemetry values
        dist = data["distances"][idx]
        zem = data["zem_norms"][idx]
        vc = data["closing_vels"][idx]
        tgo = max(0, dist / max(vc, 1))
        gload = data["g_loads"][idx]
        t = data["times"][idx]
        
        phase = "TERMINAL" if dist < 2000 else "MIDCOURSE" if dist < 5000 else "BOOST"
        
        dist_color = "#f85149" if dist < 500 else "#d29922" if dist < 2000 else "#c9d1d9"
        zem_color = "#3fb950" if zem < 50 else "#d29922" if zem < 200 else "#f85149"
        
        values = [
            (f"{dist:,.0f} m", dist_color),
            (f"{zem:,.0f} m", zem_color),
            (f"{vc:,.0f} m/s", "#c9d1d9"),
            (f"{tgo:.2f} s", "#c9d1d9"),
            (f"{gload:.1f} G", "#3fb950" if gload < 25 else "#f85149"),
            (phase, "#f85149" if phase == "TERMINAL" else "#d29922" if phase == "MIDCOURSE" else "#58a6ff"),
            (f"{t:.2f} s", "#8b949e"),
        ]
        
        for txt, (val, color) in zip(value_texts, values):
            txt.set_text(val)
            txt.set_color(color)
        
        # Show result on last frame
        if frame_num == len(frame_indices) - 1:
            if data["hit"]:
                result_text.set_text("◉ INTERCEPT")
                result_text.set_color("#3fb950")
            else:
                result_text.set_text(f"✗ MISS ({data['min_distance']:.0f}m)")
                result_text.set_color("#f85149")
        
        return [int_trail, tgt_trail, int_dot, tgt_dot, los_line,
                result_text] + value_texts
    
    anim = FuncAnimation(fig, update, frames=len(frame_indices),
                          interval=1000 // fps, blit=False)
    
    output_path = os.path.join(output_dir, "engagement_animation.gif")
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=100,
              savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    
    print(f"    ✅ Animation saved: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════
#  2. TOP-DOWN ENGAGEMENT HEATMAP
# ═══════════════════════════════════════════════════════════════

def generate_engagement_heatmap(env, n_episodes: int = 8, output_dir: str = "output"):
    """
    Create a top-down heatmap showing interceptor density
    with an infrared/thermal camera aesthetic.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"    Collecting data from {n_episodes} engagements...")
    
    all_ipos = []
    all_tpos = []
    all_closest = []
    
    for i in range(n_episodes):
        data = collect_engagement_data(env)
        all_ipos.append(data["interceptor_pos"][:, :2])  # X, Y only
        all_tpos.append(data["target_pos"][:, :2])
        
        min_idx = np.argmin(data["distances"])
        all_closest.append({
            "ipos": data["interceptor_pos"][min_idx, :2],
            "tpos": data["target_pos"][min_idx, :2],
            "dist": data["min_distance"],
            "hit": data["hit"],
        })
        status = "HIT" if data["hit"] else f"MISS ({data['min_distance']:.0f}m)"
        print(f"      [{i+1}/{n_episodes}] {status}")
    
    # Combine all interceptor positions
    all_ipos_combined = np.vstack(all_ipos)
    all_tpos_combined = np.vstack(all_tpos)
    
    # ── Create figure ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("PROJECT INTERCEPTOR — ENGAGEMENT DENSITY MAP",
                 fontsize=14, fontweight="bold", color="#58a6ff", y=0.97)
    fig.text(0.5, 0.935, f"{n_episodes} engagements overlaid  │  Infrared spectrum",
             ha="center", fontsize=10, color="#8b949e")
    
    # ── Left: Interceptor heatmap ──
    ax1 = axes[0]
    ax1.set_facecolor("#000000")
    
    # Create 2D histogram for heatmap
    x_range = [all_ipos_combined[:, 0].min() - 200, all_ipos_combined[:, 0].max() + 200]
    y_range = [all_ipos_combined[:, 1].min() - 200, all_ipos_combined[:, 1].max() + 200]
    
    heatmap, xedges, yedges = np.histogram2d(
        all_ipos_combined[:, 0], all_ipos_combined[:, 1],
        bins=80, range=[x_range, y_range]
    )
    
    # Apply gaussian smoothing manually (simple kernel)
    from scipy.ndimage import gaussian_filter
    heatmap_smooth = gaussian_filter(heatmap.T, sigma=2.5)
    
    # Custom infrared colormap
    ir_colors = ["#000000", "#1a0033", "#330066", "#660066",
                 "#990033", "#cc3300", "#ff6600", "#ffaa00",
                 "#ffdd00", "#ffffff"]
    ir_cmap = mcolors.LinearSegmentedColormap.from_list("infrared", ir_colors, N=256)
    
    im = ax1.imshow(
        heatmap_smooth,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin="lower",
        cmap=ir_cmap,
        aspect="auto",
        interpolation="bilinear",
    )
    
    # Overlay target trajectories
    for tpos in all_tpos:
        ax1.plot(tpos[:, 0], tpos[:, 1], color="#ff3333", linewidth=0.5, alpha=0.4)
    
    # Mark closest approach points
    for cp in all_closest:
        color = "#00ff00" if cp["hit"] else "#ff4444"
        ax1.scatter(cp["ipos"][0], cp["ipos"][1], color=color, s=40,
                    marker="x", linewidths=1.5, zorder=5)
    
    ax1.set_xlabel("X (downrange) [m]")
    ax1.set_ylabel("Y (crossrange) [m]")
    ax1.set_title("INTERCEPTOR DENSITY", fontsize=11, fontweight="bold",
                   color="#ff6600", pad=10)
    
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Dwell density", color="#8b949e")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    
    # ── Right: Engagement zone overlay ──
    ax2 = axes[1]
    ax2.set_facecolor("#0a0e14")
    
    for i, (ip, tp) in enumerate(zip(all_ipos, all_tpos)):
        alpha = 0.6
        ax2.plot(ip[:, 0], ip[:, 1], color=INTERCEPTOR_COLOR, linewidth=0.8, alpha=alpha)
        ax2.plot(tp[:, 0], tp[:, 1], color=TARGET_COLOR, linewidth=0.8, alpha=alpha * 0.7,
                 linestyle="--")
    
    # Mark start/end points
    for ip, tp in zip(all_ipos, all_tpos):
        ax2.scatter(ip[0, 0], ip[0, 1], color=INTERCEPTOR_COLOR, s=20, alpha=0.4, marker="^")
        ax2.scatter(tp[0, 0], tp[0, 1], color=TARGET_COLOR, s=20, alpha=0.4, marker="D")
    
    # Mark closest approach with circles
    for cp in all_closest:
        color = ACCENT_CYAN if cp["hit"] else TARGET_COLOR
        circle = Circle(
            (cp["ipos"][0], cp["ipos"][1]),
            radius=cp["dist"],
            fill=False, edgecolor=color, linewidth=1, alpha=0.5, linestyle="--"
        )
        ax2.add_patch(circle)
        ax2.scatter(cp["ipos"][0], cp["ipos"][1], color=color, s=50,
                    marker="*", zorder=5, edgecolors="white", linewidths=0.5)
    
    ax2.set_xlabel("X (downrange) [m]")
    ax2.set_ylabel("Y (crossrange) [m]")
    ax2.set_title("TRAJECTORY OVERLAY", fontsize=11, fontweight="bold",
                   color=INTERCEPTOR_COLOR, pad=10)
    ax2.set_aspect("auto")
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=INTERCEPTOR_COLOR, linewidth=1.5, label="Interceptor"),
        Line2D([0], [0], color=TARGET_COLOR, linewidth=1.5, linestyle="--", label="Target"),
        Line2D([0], [0], marker="*", color=ACCENT_CYAN, linestyle="None", markersize=8, label="Hit"),
        Line2D([0], [0], marker="*", color=TARGET_COLOR, linestyle="None", markersize=8, label="Miss"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    output_path = os.path.join(output_dir, "engagement_heatmap.png")
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    
    print(f"    ✅ Heatmap saved: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════
#  3. BEFORE / AFTER TRAINING COMPARISON
# ═══════════════════════════════════════════════════════════════

def _run_with_model(env, model, n_episodes: int = 5):
    """Run episodes using a trained model and collect data."""
    results = []
    for _ in range(n_episodes):
        data = {
            "interceptor_pos": [], "target_pos": [],
            "distances": [], "zem_norms": [],
            "g_loads": [], "times": [],
        }
        obs, _ = env.reset()
        
        # For recurrent models, we need to track LSTM states
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        done = False
        
        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states,
                                                 episode_start=episode_start,
                                                 deterministic=True)
            episode_start = np.zeros((1,), dtype=bool)
            obs, reward, terminated, truncated, info = env.step(action)
            raw = env.get_raw_state()
            done = terminated or truncated
            
            data["interceptor_pos"].append(raw["interceptor_pos"])
            data["target_pos"].append(raw["target_pos"])
            data["distances"].append(raw["distance"])
            data["zem_norms"].append(raw["zem_norm"])
            data["g_loads"].append(np.linalg.norm(action) / 9.81)
            data["times"].append(raw["time_seconds"])
        
        stats = info.get("episode_stats", {})
        for key in data:
            data[key] = np.array(data[key])
        data["hit"] = stats.get("hit", False)
        data["min_distance"] = stats.get("min_distance", float("inf"))
        results.append(data)
    
    return results


def generate_before_after_comparison(env, model_path: str = None, 
                                      n_episodes: int = 5, output_dir: str = "output"):
    """
    Generate a side-by-side comparison of random vs trained agent.
    If no trained model exists, generates random vs PN (proportional navigation).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # ── Collect "BEFORE" data (random agent) ──
    print("    Collecting BEFORE data (random agent)...")
    random_data = []
    for i in range(n_episodes):
        d = collect_engagement_data(env)
        random_data.append(d)
        print(f"      Random [{i+1}/{n_episodes}]: {'HIT' if d['hit'] else 'MISS'} | "
              f"Min: {d['min_distance']:.0f}m")
    
    # ── Collect "AFTER" data ──
    trained_data = []
    model_loaded = False
    
    # Try to load a trained model
    if model_path is None:
        model_path = "models/interceptor_lstm_guidance"
    
    try:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path)
        model_loaded = True
        print(f"    Collecting AFTER data (trained model: {model_path})...")
        trained_data = _run_with_model(env, model, n_episodes)
        for i, d in enumerate(trained_data):
            print(f"      Trained [{i+1}/{n_episodes}]: {'HIT' if d['hit'] else 'MISS'} | "
                  f"Min: {d['min_distance']:.0f}m")
    except Exception as e:
        print(f"    No trained model found ({e})")
        print("    Collecting AFTER data (proportional navigation baseline)...")
        
        # Fallback: Use PN guidance as "after"
        from simulation.kinematics import proportional_navigation, compute_los_rate
        for i in range(n_episodes):
            data = {
                "interceptor_pos": [], "target_pos": [],
                "distances": [], "zem_norms": [],
                "g_loads": [], "times": [],
            }
            obs, _ = env.reset()
            done = False
            while not done:
                raw = env.get_raw_state()
                rel_pos = np.array(raw["relative_position"])
                rel_vel = np.array(raw["relative_velocity"])
                
                # PN guidance: compute acceleration command
                los_rate = compute_los_rate(rel_pos, rel_vel)
                vc = raw["closing_velocity"]
                N = 4.0  # navigation constant
                a_cmd = N * vc * los_rate
                
                # Map 3D command to 2D action (y, z components)
                action = np.clip(a_cmd[1:3], -30 * 9.81, 30 * 9.81).astype(np.float32)
                
                obs, reward, terminated, truncated, info = env.step(action)
                raw = env.get_raw_state()
                done = terminated or truncated
                
                data["interceptor_pos"].append(raw["interceptor_pos"])
                data["target_pos"].append(raw["target_pos"])
                data["distances"].append(raw["distance"])
                data["zem_norms"].append(raw["zem_norm"])
                data["g_loads"].append(np.linalg.norm(action) / 9.81)
                data["times"].append(raw["time_seconds"])
            
            stats = info.get("episode_stats", {})
            for key in data:
                data[key] = np.array(data[key])
            data["hit"] = stats.get("hit", False)
            data["min_distance"] = stats.get("min_distance", float("inf"))
            trained_data.append(data)
            print(f"      PN [{i+1}/{n_episodes}]: {'HIT' if data['hit'] else 'MISS'} | "
                  f"Min: {data['min_distance']:.0f}m")
    
    after_label = "TRAINED AGENT" if model_loaded else "PROPORTIONAL NAV (N=4)"
    
    # ── Create comparison figure ──
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("PROJECT INTERCEPTOR — BEFORE / AFTER COMPARISON",
                 fontsize=15, fontweight="bold", color="#58a6ff", y=0.98)
    
    random_hits = sum(1 for d in random_data if d["hit"])
    trained_hits = sum(1 for d in trained_data if d["hit"])
    random_avg_min = np.mean([d["min_distance"] for d in random_data])
    trained_avg_min = np.mean([d["min_distance"] for d in trained_data])
    
    fig.text(0.5, 0.955,
             f"Random: {random_hits}/{n_episodes} hits (avg {random_avg_min:.0f}m)  │  "
             f"{after_label}: {trained_hits}/{n_episodes} hits (avg {trained_avg_min:.0f}m)",
             ha="center", fontsize=10, color="#8b949e")
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.06, right=0.97, top=0.92, bottom=0.06)
    
    # ── Top left: Random trajectories ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#0a0e14")
    for d in random_data:
        color = ACCENT_CYAN if d["hit"] else TARGET_COLOR
        ax1.plot(d["interceptor_pos"][:, 0], d["interceptor_pos"][:, 1],
                 color=color, alpha=0.6, linewidth=1)
        ax1.plot(d["target_pos"][:, 0], d["target_pos"][:, 1],
                 color="#666666", alpha=0.3, linewidth=0.8, linestyle=":")
    
    ax1.set_title(f"BEFORE: RANDOM AGENT ({random_hits}/{n_episodes} hits)",
                   fontsize=11, fontweight="bold", color=TARGET_COLOR, pad=10)
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    
    # ── Top right: Trained trajectories ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#0a0e14")
    for d in trained_data:
        color = ACCENT_CYAN if d["hit"] else TARGET_COLOR
        ax2.plot(d["interceptor_pos"][:, 0], d["interceptor_pos"][:, 1],
                 color=color, alpha=0.6, linewidth=1)
        ax2.plot(d["target_pos"][:, 0], d["target_pos"][:, 1],
                 color="#666666", alpha=0.3, linewidth=0.8, linestyle=":")
    
    ax2.set_title(f"AFTER: {after_label} ({trained_hits}/{n_episodes} hits)",
                   fontsize=11, fontweight="bold", color=ACCENT_CYAN, pad=10)
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Y [m]")
    
    # ── Bottom left: ZEM comparison ──
    ax3 = fig.add_subplot(gs[1, 0])
    for d in random_data:
        ax3.plot(d["times"], d["zem_norms"], color=TARGET_COLOR, alpha=0.3, linewidth=0.8)
    for d in trained_data:
        ax3.plot(d["times"], d["zem_norms"], color=ACCENT_CYAN, alpha=0.5, linewidth=0.8)
    
    # Mean lines
    max_t_random = max(d["times"][-1] for d in random_data)
    max_t_trained = max(d["times"][-1] for d in trained_data)
    ax3.axhline(y=np.mean([d["zem_norms"].mean() for d in random_data]),
                color=TARGET_COLOR, linewidth=2, linestyle="--", alpha=0.8, label=f"Random avg")
    ax3.axhline(y=np.mean([d["zem_norms"].mean() for d in trained_data]),
                color=ACCENT_CYAN, linewidth=2, linestyle="--", alpha=0.8, label=f"{after_label} avg")
    
    ax3.set_ylabel("ZEM [m]")
    ax3.set_xlabel("Time [s]")
    ax3.set_title("ZERO-EFFORT-MISS COMPARISON", fontsize=11, fontweight="bold",
                   color=ZEM_COLOR, pad=10)
    ax3.legend(fontsize=8)
    
    # ── Bottom right: Min distance bar comparison ──
    ax4 = fig.add_subplot(gs[1, 1])
    
    x_pos = np.arange(n_episodes)
    width = 0.35
    
    random_mins = [d["min_distance"] for d in random_data]
    trained_mins = [d["min_distance"] for d in trained_data]
    
    bars1 = ax4.bar(x_pos - width/2, random_mins, width, color=TARGET_COLOR,
                     alpha=0.7, label="Random", edgecolor="#30363d")
    bars2 = ax4.bar(x_pos + width/2, trained_mins, width, color=ACCENT_CYAN,
                     alpha=0.7, label=after_label, edgecolor="#30363d")
    
    ax4.axhline(y=5.0, color="#ffffff", linewidth=1, linestyle="--", alpha=0.4, label="Kill radius (5m)")
    
    # Value labels
    for bar, val in zip(bars1, random_mins):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                 f"{val:.0f}", ha="center", fontsize=7, color=TARGET_COLOR)
    for bar, val in zip(bars2, trained_mins):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                 f"{val:.0f}", ha="center", fontsize=7, color=ACCENT_CYAN)
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"ENG-{i+1}" for i in range(n_episodes)])
    ax4.set_ylabel("Min Distance [m]")
    ax4.set_title("CLOSEST APPROACH COMPARISON", fontsize=11, fontweight="bold",
                   color=CLOSING_COLOR, pad=10)
    ax4.legend(fontsize=8)
    
    output_path = os.path.join(output_dir, "before_after_comparison.png")
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    
    print(f"    ✅ Comparison saved: {output_path}")
    return output_path
