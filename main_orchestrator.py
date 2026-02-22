"""
Project INTERCEPTOR — Main Orchestrator
======================================

Entry point that ties all five microservice layers together.
Provides CLI modes for running different combinations of layers.

Modes:
    --mode sim      Run the physics simulation only (no RL, no Kafka)
    --mode demo     Rich visual engagement display (impressive output)
    --mode train    Train the LSTM-PPO agent
    --mode stream   Sim + Kafka telemetry streaming
    --mode filter   Test the causal AI decoy filter
    --mode graph    Test the LangGraph mission supervisor
    --mode full     All layers active (requires Kafka + dependencies)
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("interceptor")


BANNER = """
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║     ██╗███╗   ██╗████████╗███████╗██████╗  ██████╗███████╗     ║
║     ██║████╗  ██║╚══██╔══╝██╔════╝██╔══██╗██╔════╝██╔════╝     ║
║     ██║██╔██╗ ██║   ██║   █████╗  ██████╔╝██║     █████╗       ║
║     ██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗██║     ██╔══╝       ║
║     ██║██║ ╚████║   ██║   ███████╗██║  ██║╚██████╗███████╗     ║
║     ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝ ╚═════╝╚══════╝     ║
║               ██████╗ ████████╗ ██████╗ ██████╗                ║
║               ██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗               ║
║               ██████╔╝   ██║   ██║   ██║██████╔╝               ║
║               ██╔═══╝    ██║   ██║   ██║██╔══██╗               ║
║               ██║        ██║   ╚██████╔╝██║  ██║               ║
║               ╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝               ║
║                                                                ║
║          Autonomous Swarm Defense Architecture                 ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""


def run_simulation_only(n_episodes: int = 5, render: bool = False):
    """
    Mode: sim — Run the physics simulation standalone.
    Tests the Gymnasium environment with random actions.
    """
    from simulation.guidance_env import GuidanceEnv
    
    print("\n── Mode: SIMULATION ONLY ──")
    env = GuidanceEnv()
    
    total_hits = 0
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        stats = info.get("episode_stats", {})
        hit = stats.get("hit", False)
        total_hits += int(hit)
        
        print(
            f"  Episode {ep+1}/{n_episodes}: "
            f"{'HIT ✓' if hit else 'MISS ✗'} | "
            f"Steps: {steps} | "
            f"Min dist: {stats.get('min_distance', -1):.2f}m | "
            f"Reward: {total_reward:.2f}"
        )
    
    print(f"\n  Summary: {total_hits}/{n_episodes} hits ({total_hits/n_episodes*100:.0f}%)")
    env.close()


def run_demo_mode(n_episodes: int = 3):
    """Mode: demo — Rich visual real-time engagement display."""
    from simulation.guidance_env import GuidanceEnv
    from simulation.visualizer import run_demo
    
    env = GuidanceEnv()
    run_demo(env, n_episodes=n_episodes)
    env.close()


def run_plot_mode(n_episodes: int = 5):
    """Mode: plot — Generate all matplotlib visualizations."""
    from simulation.guidance_env import GuidanceEnv
    from simulation.plotter import generate_engagement_plot, generate_multi_engagement_plot
    from simulation.advanced_viz import (
        generate_animated_engagement,
        generate_engagement_heatmap,
        generate_before_after_comparison,
    )
    
    print("\n── Mode: PLOT GENERATION ──")
    env = GuidanceEnv()
    
    # 1. Single engagement detailed plot
    print("\n  [1/5] Single engagement trajectory...")
    generate_engagement_plot(env, output_dir="output")
    
    # 2. Multi-engagement comparison
    print(f"\n  [2/5] {n_episodes}-engagement comparison...")
    generate_multi_engagement_plot(env, n_episodes=n_episodes, output_dir="output")
    
    # 3. Animated engagement GIF
    print("\n  [3/5] Animated engagement GIF...")
    generate_animated_engagement(env, output_dir="output")
    
    # 4. Engagement heatmap (infrared)
    print("\n  [4/5] Engagement heatmap (infrared)...")
    generate_engagement_heatmap(env, n_episodes=max(n_episodes, 8), output_dir="output")
    
    # 5. Before/After comparison
    print("\n  [5/5] Before/After comparison...")
    generate_before_after_comparison(env, n_episodes=n_episodes, output_dir="output")
    
    env.close()
    print("\n  ✅ All plots saved to output/ directory!")
    print("  Files: engagement_plot.png, multi_engagement.png,")
    print("         engagement_animation.gif, engagement_heatmap.png,")
    print("         before_after_comparison.png")


def run_training(timesteps: int = 500_000):
    """Mode: train — Train the LSTM-PPO agent."""
    from intelligence.train_agent import train_guidance_agent
    
    print("\n── Mode: TRAINING ──")
    train_guidance_agent({"total_timesteps": timesteps})


def run_streaming_simulation(n_steps: int = 500, enable_kafka: bool = False):
    """
    Mode: stream — Run simulation with Kafka telemetry streaming.
    Falls back to in-memory if Kafka is unavailable.
    """
    from simulation.guidance_env import GuidanceEnv
    from streaming.producer import TelemetryStreamer
    from streaming.consumer import TelemetryConsumer
    
    print("\n── Mode: STREAMING SIMULATION ──")
    
    env = GuidanceEnv()
    streamer = TelemetryStreamer(enable_kafka=enable_kafka)
    consumer = TelemetryConsumer(enable_kafka=enable_kafka)
    
    obs, _ = env.reset()
    
    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Stream raw state to Kafka (or buffer)
        raw_state = env.get_raw_state()
        streamer.stream_state(raw_state, interceptor_id=0)
        
        if step % 100 == 0:
            print(
                f"  Step {step}: distance={raw_state['distance']:.1f}m, "
                f"ZEM={raw_state['zem_norm']:.1f}m, "
                f"V_c={raw_state['closing_velocity']:.1f}m/s"
            )
        
        if terminated or truncated:
            event_type = "INTERCEPT_HIT" if info.get("episode_stats", {}).get("hit") else "INTERCEPT_MISS"
            streamer.stream_event(event_type, info.get("episode_stats", {}))
            break
    
    # In non-Kafka mode, feed buffer to consumer
    if not enable_kafka:
        consumer.ingest_from_buffer(streamer.get_buffer())
        latest = consumer.get_latest_state(0)
        if latest:
            print(f"\n  Consumer received latest state: time_step={latest.get('time_step')}")
        
        events = consumer.get_latest_events()
        if events:
            print(f"  Events: {[e.get('event_type') for e in events]}")
    
    print(f"\n  Total messages streamed: {len(streamer.get_buffer())}")
    streamer.close()
    env.close()


def run_causal_filter_test():
    """Mode: filter — Test the causal AI decoy rejection filter."""
    from intelligence.causal_filter import CausalDecoyFilter
    
    print("\n── Mode: CAUSAL FILTER TEST ──")
    
    cf = CausalDecoyFilter()
    
    # Scenario: 3 tracks — 1 real target, 1 decoy (high G), 1 decoy (RCS decay)
    test_tracks = [
        {"id": 1, "accel": 7.0, "velocity": [300, 10, 5], "position": [5000, 100, 50],
         "label": "Real target (7G maneuver)"},
        {"id": 2, "accel": 180.0, "velocity": [300, 0, 0], "position": [5000, 120, 50],
         "label": "Decoy flare (180G impossible)"},
        {"id": 3, "accel": 50.0, "velocity": [300, 0, 0], "position": [5000, 80, 50],
         "label": "Ambiguous track (50G — edge case)"},
    ]
    
    print("  Evaluating tracks:")
    results = cf.evaluate_tracks(test_tracks)
    
    for track, result in zip(test_tracks, results):
        status = "🚫 DECOY" if result["is_decoy"] else "✅ REAL"
        print(f"    Track {track['id']} ({track['label']}): {status}")
        print(f"      → {result['reason']}")
    
    print(f"\n  Filter stats: {cf.get_stats()}")


def run_graph_test():
    """Mode: graph — Test the LangGraph mission supervisor."""
    from supervisor.graph_state import create_initial_mission_state
    from supervisor.vlm_commander import build_mission_graph
    
    print("\n── Mode: LANGGRAPH SUPERVISOR TEST ──")
    
    graph = build_mission_graph()
    print(f"  ✅ Graph compiled: {type(graph).__name__}")
    
    # Create a scenario where interceptor 0 hits the target
    state = create_initial_mission_state("ORCH-001", n_interceptors=3, n_targets=1)
    state["interceptors"][0]["distance_to_target"] = 3.0
    state["targets"][0]["decoy_confidence"] = 0.1
    
    print("\n  Running engagement scenario...")
    
    for step_output in graph.stream(state):
        node_name = list(step_output.keys())[0]
        result = step_output[node_name]
        print(f"    [{node_name}] → {result.get('supervisor_action', 'N/A')}")
    
    print(f"\n  ✅ Mission completed. Phase: {result.get('phase', 'unknown')}")


def run_full_system(timesteps: int = 10_000, enable_kafka: bool = False):
    """
    Mode: full — Run all layers together.
    
    1. Initialize simulation
    2. Initialize Kafka streaming
    3. Initialize causal filter
    4. Initialize LangGraph supervisor
    5. Run integrated loop
    """
    from simulation.guidance_env import GuidanceEnv
    from streaming.producer import TelemetryStreamer
    from streaming.consumer import TelemetryConsumer
    from intelligence.causal_filter import CausalDecoyFilter
    from supervisor.graph_state import create_initial_mission_state
    from supervisor.vlm_commander import build_mission_graph
    
    print("\n── Mode: FULL SYSTEM INTEGRATION ──")
    
    # Layer 1: Physics
    env = GuidanceEnv()
    print("  ✅ Physics engine initialized")
    
    # Layer 2: Streaming
    streamer = TelemetryStreamer(enable_kafka=enable_kafka)
    consumer = TelemetryConsumer(enable_kafka=enable_kafka)
    print("  ✅ Telemetry streaming initialized")
    
    # Layer 3: Causal filter
    causal_filter = CausalDecoyFilter()
    print("  ✅ Causal AI filter initialized")
    
    # Layer 4: Supervisor
    graph = build_mission_graph()
    mission_state = create_initial_mission_state("FULL-001", n_interceptors=1)
    print("  ✅ LangGraph supervisor initialized")
    
    print(f"\n  Running integrated loop for up to {timesteps} steps...")
    
    obs, _ = env.reset()
    episode = 0
    total_hits = 0
    
    for step in range(timesteps):
        # RL action (random for now — would use trained model)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Stream to Kafka / buffer
        raw_state = env.get_raw_state()
        streamer.stream_state(raw_state)
        
        # Run causal filter on the track
        track_report = {
            "id": 0,
            "accel": np.linalg.norm(raw_state.get("target_accel", [0, 0, 0])) / 9.81,
            "velocity": raw_state.get("relative_velocity", [0, 0, 0]),
            "position": raw_state.get("relative_position", [0, 0, 0]),
        }
        filter_result = causal_filter.evaluate_tracks([track_report])
        
        # Update supervisor state
        mission_state["interceptors"][0]["distance_to_target"] = raw_state["distance"]
        mission_state["interceptors"][0]["zem_norm"] = raw_state["zem_norm"]
        mission_state["interceptors"][0]["closing_velocity"] = raw_state["closing_velocity"]
        mission_state["interceptors"][0]["t_go"] = raw_state["t_go"]
        mission_state["targets"][0]["decoy_confidence"] = (
            1.0 - filter_result[0]["confidence"] if filter_result[0]["is_decoy"] else 0.0
        )
        
        if step % 200 == 0:
            print(
                f"  Step {step}: dist={raw_state['distance']:.0f}m, "
                f"ZEM={raw_state['zem_norm']:.0f}m, "
                f"filter={'DECOY' if filter_result[0]['is_decoy'] else 'OK'}"
            )
        
        if terminated or truncated:
            hit = info.get("episode_stats", {}).get("hit", False)
            total_hits += int(hit)
            is_hit_str = "HIT ✓" if hit else "MISS ✗"
            
            # Run supervisor for terminal assessment
            if hit:
                mission_state["interceptors"][0]["distance_to_target"] = 3.0
            else:
                # Mark interceptor as missed so the graph terminates
                mission_state["interceptors"][0]["status"] = "missed"
                mission_state["interceptors"][0]["closing_velocity"] = -500.0
                mission_state["phase"] = "complete"
            
            # Stream supervisor cycle (bounded recursion to prevent infinite loops)
            try:
                for step_output in graph.stream(
                    dict(mission_state),
                    {"recursion_limit": 15}
                ):
                    pass  # Process supervisor nodes
            except Exception as e:
                logger.debug(f"Supervisor cycle ended: {e}")
            
            event_type = "INTERCEPT_HIT" if hit else "INTERCEPT_MISS"
            streamer.stream_event(event_type, info.get("episode_stats", {}))
            
            episode += 1
            print(f"    Episode {episode}: {is_hit_str} | "
                  f"Min dist: {info.get('episode_stats', {}).get('min_distance', -1):.2f}m")
            
            if episode >= 5:
                break
            
            obs, _ = env.reset()
            mission_state = create_initial_mission_state(f"FULL-{episode:03d}", n_interceptors=1)
    
    # Final stats
    if not enable_kafka:
        consumer.ingest_from_buffer(streamer.get_buffer())
    
    print(f"\n  ═══ FULL SYSTEM RESULTS ═══")
    print(f"  Episodes:        {episode}")
    print(f"  Hits:            {total_hits}/{episode}")
    print(f"  Messages:        {len(streamer.get_buffer())}")
    print(f"  Filter stats:    {causal_filter.get_stats()}")
    
    streamer.close()
    env.close()


def main():
    print(BANNER)
    
    parser = argparse.ArgumentParser(
        description="Project INTERCEPTOR — Autonomous Swarm Defense Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "demo", "plot", "train", "stream", "filter", "graph", "full"],
        default="sim",
        help="Operation mode (default: sim)"
    )
    parser.add_argument("--timesteps", type=int, default=500_000, help="Training timesteps")
    parser.add_argument("--episodes", type=int, default=5, help="Simulation episodes")
    parser.add_argument("--kafka", action="store_true", help="Enable Kafka (requires running cluster)")
    
    args = parser.parse_args()
    
    mode_map = {
        "sim": lambda: run_simulation_only(args.episodes),
        "demo": lambda: run_demo_mode(args.episodes),
        "plot": lambda: run_plot_mode(args.episodes),
        "train": lambda: run_training(args.timesteps),
        "stream": lambda: run_streaming_simulation(enable_kafka=args.kafka),
        "filter": run_causal_filter_test,
        "graph": run_graph_test,
        "full": lambda: run_full_system(enable_kafka=args.kafka),
    }
    
    try:
        mode_map[args.mode]()
    except KeyboardInterrupt:
        print("\n\n  Mission aborted by operator.")
    except Exception as e:
        logger.exception(f"Fatal error in mode '{args.mode}': {e}")
        raise


if __name__ == "__main__":
    main()
