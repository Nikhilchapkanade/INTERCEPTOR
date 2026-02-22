"""
Project INTERCEPTOR — Real-Time Engagement Visualizer
=====================================================

Rich terminal visualization for the missile interception engagement.
Uses the `rich` library for colored, animated, production-grade output
that looks impressive in demos and recordings.
"""

import time
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.columns import Columns
from rich import box


console = Console()


def create_radar_display(
    interceptor_pos: list,
    target_pos: list,
    width: int = 40,
    height: int = 18
) -> str:
    """
    Create an ASCII radar display showing interceptor and target positions.
    Projects 3D positions onto a 2D XY radar plane.
    """
    grid = [[" " for _ in range(width)] for _ in range(height)]
    
    # Draw border
    for x in range(width):
        grid[0][x] = "─"
        grid[height - 1][x] = "─"
    for y in range(height):
        grid[y][0] = "│"
        grid[y][width - 1] = "│"
    grid[0][0] = "┌"
    grid[0][width - 1] = "┐"
    grid[height - 1][0] = "└"
    grid[height - 1][width - 1] = "┘"
    
    # Draw crosshairs
    mid_x, mid_y = width // 2, height // 2
    for x in range(2, width - 2):
        if grid[mid_y][x] == " ":
            grid[mid_y][x] = "·"
    for y in range(2, height - 2):
        if grid[y][mid_x] == " ":
            grid[y][mid_x] = "·"
    grid[mid_y][mid_x] = "┼"
    
    # Range rings
    for angle in range(0, 360, 30):
        rad = np.radians(angle)
        for r_factor in [0.3, 0.6, 0.9]:
            rx = int(mid_x + r_factor * (mid_x - 2) * np.cos(rad))
            ry = int(mid_y + r_factor * (mid_y - 2) * np.sin(rad))
            if 1 < rx < width - 1 and 1 < ry < height - 1:
                if grid[ry][rx] == " ":
                    grid[ry][rx] = "∘"
    
    # Map positions to grid (X axis = horizontal, Y axis = vertical)
    max_range = max(abs(target_pos[0] - interceptor_pos[0]) + 1000, 5000)
    
    # Interceptor position (always centered initially)
    ix = int(mid_x + (interceptor_pos[0] / max_range) * (mid_x - 3))
    iy = int(mid_y - (interceptor_pos[1] / max_range) * (mid_y - 3))
    ix = max(2, min(width - 3, ix))
    iy = max(2, min(height - 3, iy))
    
    # Target position
    tx = int(mid_x + (target_pos[0] / max_range) * (mid_x - 3))
    ty = int(mid_y - (target_pos[1] / max_range) * (mid_y - 3))
    tx = max(2, min(width - 3, tx))
    ty = max(2, min(height - 3, ty))
    
    # Draw trail between interceptor and target
    steps = max(abs(tx - ix), abs(ty - iy), 1)
    for s in range(1, steps):
        px = ix + int((tx - ix) * s / steps)
        py = iy + int((ty - iy) * s / steps)
        if 1 < px < width - 1 and 1 < py < height - 1:
            if grid[py][px] in (" ", "·", "∘"):
                grid[py][px] = "░"
    
    # Place interceptor and target markers
    if 1 < ix < width - 1 and 1 < iy < height - 1:
        grid[iy][ix] = "▲"
    if 1 < tx < width - 1 and 1 < ty < height - 1:
        grid[ty][tx] = "◆"
    
    return "\n".join("".join(row) for row in grid)


def get_threat_bar(value: float, max_val: float, width: int = 20) -> str:
    """Create a colored progress bar."""
    ratio = min(value / max(max_val, 1), 1.0)
    filled = int(ratio * width)
    empty = width - filled
    
    if ratio < 0.3:
        color = "green"
    elif ratio < 0.7:
        color = "yellow"
    else:
        color = "red"
    
    bar = f"[{color}]{'█' * filled}{'░' * empty}[/{color}]"
    return bar


def run_demo(env, n_episodes: int = 3):
    """
    Run a visually impressive real-time engagement demo.
    
    This creates a rich terminal display showing:
    - ASCII radar plot
    - Live telemetry gauges
    - Engagement statistics
    - Causal filter status
    - Mission log
    """
    from intelligence.causal_filter import CausalDecoyFilter
    
    causal_filter = CausalDecoyFilter()
    
    total_hits = 0
    episode_results = []
    
    console.print()
    console.print(Panel(
        "[bold cyan]INTERCEPTOR DEFENSE SYSTEM[/bold cyan]\n"
        "[dim]Autonomous Swarm Interception — Real-Time Engagement Monitor[/dim]",
        border_style="cyan",
        box=box.DOUBLE_EDGE,
        padding=(1, 4),
    ))
    console.print()
    time.sleep(1)
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        episode_log = []
        
        console.print(f"\n[bold yellow]━━━ ENGAGEMENT {ep + 1}/{n_episodes} INITIALIZED ━━━[/bold yellow]")
        time.sleep(0.5)
        
        with Live(console=console, refresh_per_second=6, transient=True) as live:
          try:
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                raw = env.get_raw_state()
                step_count += 1
                done = terminated or truncated
                
                # Run causal filter every 50 steps
                filter_status = "NOMINAL"
                if step_count % 50 == 0:
                    track_report = {
                        "id": 0,
                        "accel": np.linalg.norm(raw.get("target_accel", [0, 0, 0])) / 9.81,
                        "velocity": raw.get("relative_velocity", [0, 0, 0]),
                        "position": raw.get("relative_position", [0, 0, 0]),
                    }
                    result = causal_filter.evaluate_tracks([track_report])
                    if result[0]["is_decoy"]:
                        filter_status = "⚠ DECOY DETECTED"
                
                # Only render every 15 steps for performance
                if step_count % 15 != 0 and not done:
                    continue
                
                distance = raw["distance"]
                zem = raw["zem_norm"]
                vc = raw["closing_velocity"]
                tgo = raw["t_go"]
                
                # ─── Build the display ───
                
                # Radar panel
                radar_str = create_radar_display(
                    raw["interceptor_pos"],
                    raw["target_pos"],
                    width=42,
                    height=16
                )
                
                # Phase indicator
                phase = "TERMINAL" if distance < 2000 else "MIDCOURSE" if distance < 5000 else "BOOST"
                phase_color = "red" if phase == "TERMINAL" else "yellow" if phase == "MIDCOURSE" else "cyan"
                
                g_load = np.linalg.norm(action) / 9.81
                dist_color = "red" if distance < 500 else "yellow" if distance < 2000 else "white"
                zem_color = "green" if zem < 50 else "yellow" if zem < 200 else "red"
                
                display_text = (
                    f"[green]{radar_str}[/green]\n"
                    f"  [dim]▲ Interceptor   ◆ Target   ░ LOS[/dim]\n"
                    f"\n"
                    f"  [{phase_color}]■ {phase}[/{phase_color}]   "
                    f"Step: [bold]{step_count:,}[/bold]   "
                    f"Time: [cyan]{raw['time_seconds']:.2f}s[/cyan]   "
                    f"Filter: [green]{filter_status}[/green]\n"
                    f"\n"
                    f"  Distance:  [{dist_color}]{distance:>8,.1f} m[/{dist_color}]   "
                    f"ZEM:  [{zem_color}]{zem:>8,.1f} m[/{zem_color}]   "
                    f"Closing: {vc:>8,.1f} m/s\n"
                    f"  T-go:      {tgo:>8.2f} s   "
                    f"G-Load: {g_load:>6.1f} G    "
                    f"Cmd: [{action[0]:+.0f}, {action[1]:+.0f}] m/s²\n"
                    f"\n"
                    f"  Threat  {get_threat_bar(10000 - distance, 10000, 25)}\n"
                    f"  ZEM     {get_threat_bar(zem, 1000, 25)}\n"
                    f"  Fuel    {get_threat_bar(max(0, 100 - step_count * 0.08), 100, 25)}"
                )
                
                panel = Panel(
                    display_text,
                    title=f"[bold white on blue] ◉ ENGAGEMENT {ep+1}/{n_episodes} — LIVE [/bold white on blue]",
                    border_style="blue",
                    subtitle=f"[dim]▲ Interceptor Active  |  Track: TGT-001 LOCKED  |  Supervisor: ONLINE[/dim]",
                    width=80,
                )
                
                live.update(panel)
                
                if not done:
                    time.sleep(0.03)
          except Exception:
              pass
        
        # Episode result
        stats = info.get("episode_stats", {})
        hit = stats.get("hit", False)
        total_hits += int(hit)
        min_dist = stats.get("min_distance", float('inf'))
        
        episode_results.append({
            "episode": ep + 1,
            "hit": hit,
            "min_distance": min_dist,
            "steps": step_count,
        })
        
        if hit:
            console.print(Panel(
                f"[bold green]◉ INTERCEPT SUCCESSFUL[/bold green]\n"
                f"[green]Miss distance: {min_dist:.2f}m | Steps: {step_count}[/green]",
                border_style="green",
                box=box.DOUBLE_EDGE,
            ))
        else:
            console.print(Panel(
                f"[bold red]✗ INTERCEPT FAILED[/bold red]\n"
                f"[red]Min distance: {min_dist:.2f}m | Steps: {step_count}[/red]",
                border_style="red",
                box=box.DOUBLE_EDGE,
            ))
        
        time.sleep(1)
    
    # ─── Final Summary ───
    console.print()
    
    summary_table = Table(
        title="[bold cyan]MISSION DEBRIEF[/bold cyan]",
        box=box.DOUBLE_EDGE,
        border_style="cyan",
        show_lines=True,
        padding=(0, 2),
    )
    summary_table.add_column("Engagement", style="white", justify="center")
    summary_table.add_column("Result", justify="center")
    summary_table.add_column("Min Distance", justify="right", style="yellow")
    summary_table.add_column("Duration", justify="right", style="dim")
    
    for r in episode_results:
        result_str = "[bold green]HIT ◉[/bold green]" if r["hit"] else "[bold red]MISS ✗[/bold red]"
        summary_table.add_row(
            f"ENG-{r['episode']:03d}",
            result_str,
            f"{r['min_distance']:.2f} m",
            f"{r['steps']} steps",
        )
    
    summary_table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_hits}/{n_episodes}[/bold]",
        "",
        f"[bold]{total_hits / n_episodes * 100:.0f}% hit rate[/bold]",
    )
    
    console.print(summary_table)
    console.print()
    
    # Causal filter stats
    cf_stats = causal_filter.get_stats()
    console.print(Panel(
        f"  Tracks Analyzed: [cyan]{cf_stats['total_tracks_analyzed']}[/cyan]\n"
        f"  Decoys Rejected: [red]{cf_stats['decoys_rejected']}[/red]\n"
        f"  Accel Violations: [yellow]{cf_stats['accel_violations']}[/yellow]",
        title="[bold]CAUSAL FILTER REPORT[/bold]",
        border_style="dim",
        width=45,
    ))
    console.print()
