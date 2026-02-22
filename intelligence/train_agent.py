"""
Project INTERCEPTOR — Recurrent PPO Training Script
==================================================

Trains an LSTM-PPO agent on the GuidanceEnv using sb3-contrib's
RecurrentPPO. The LSTM hidden state allows the agent to track
the target's evasive maneuver patterns over time.

Features:
- Configurable hyperparameters
- TensorBoard logging  
- Model checkpointing
- Evaluation callback with best-model saving
- Optional Kafka telemetry streaming during training
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from simulation.guidance_env import GuidanceEnv

logger = logging.getLogger(__name__)


# ─── Hyperparameters ───
TRAINING_CONFIG = {
    # Environment
    "env_config": {
        "max_g": 30.0,
        "dt": 0.01,
        "max_steps": 1500,
        "hit_radius": 5.0,
        "initial_range": 10000.0,
        "interceptor_speed": 1000.0,
        "target_speed": 300.0,
        "target_maneuver_g": 7.0,
        "k_zem": 0.001,
        "k_accel": 1e-6,
        "r_hit": 500.0,
        "r_miss": -100.0,
    },
    
    # RecurrentPPO
    "policy": "MlpLstmPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,         # Entropy bonus for exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    
    # LSTM
    "policy_kwargs": {
        "lstm_hidden_size": 128,
        "n_lstm_layers": 2,
        "shared_lstm": False,    # Separate LSTM for actor and critic
        "enable_critic_lstm": True,
    },
    
    # Training
    "total_timesteps": 500_000,   # Reduced for quick verification
    "n_envs": 4,                   # Parallel environments
    "eval_freq": 10_000,
    "checkpoint_freq": 50_000,
    
    # Paths
    "model_dir": "models",
    "log_dir": "guidance_tensorboard",
    "model_name": "interceptor_lstm_guidance",
}


def make_env(config: dict, rank: int = 0):
    """Create a wrapped GuidanceEnv instance."""
    def _init():
        env = GuidanceEnv(config=config)
        env = Monitor(env)
        return env
    return _init


def train_guidance_agent(config: dict = None):
    """
    Train the LSTM-PPO guidance agent.
    
    Args:
        config: Override training configuration dict
    """
    cfg = {**TRAINING_CONFIG, **(config or {})}
    
    # Create output directories
    os.makedirs(cfg["model_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)
    
    print("=" * 70)
    print("  PROJECT INTERCEPTOR — LSTM-PPO Guidance Agent Training")
    print("=" * 70)
    print(f"  Total timesteps:   {cfg['total_timesteps']:,}")
    print(f"  Parallel envs:     {cfg['n_envs']}")
    print(f"  LSTM hidden size:  {cfg['policy_kwargs']['lstm_hidden_size']}")
    print(f"  LSTM layers:       {cfg['policy_kwargs']['n_lstm_layers']}")
    print(f"  Learning rate:     {cfg['learning_rate']}")
    print(f"  Batch size:        {cfg['batch_size']}")
    print("=" * 70)
    
    # ─── Create environments ───
    print("\n[1/4] Creating training environments...")
    env_fns = [make_env(cfg["env_config"], i) for i in range(cfg["n_envs"])]
    train_env = DummyVecEnv(env_fns)
    
    # Evaluation environment (single, deterministic)
    eval_env = DummyVecEnv([make_env(cfg["env_config"])])
    
    # ─── Callbacks ───
    print("[2/4] Setting up callbacks...")
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(cfg["model_dir"], "best"),
        log_path=os.path.join(cfg["log_dir"], "eval"),
        eval_freq=cfg["eval_freq"] // cfg["n_envs"],
        n_eval_episodes=10,
        deterministic=True,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg["checkpoint_freq"] // cfg["n_envs"],
        save_path=os.path.join(cfg["model_dir"], "checkpoints"),
        name_prefix=cfg["model_name"],
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # ─── Initialize model ───
    print("[3/4] Initializing RecurrentPPO model...")
    model = RecurrentPPO(
        policy=cfg["policy"],
        env=train_env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        policy_kwargs=cfg["policy_kwargs"],
        tensorboard_log=cfg["log_dir"],
        verbose=1,
        seed=42,
    )
    
    param_count = sum(p.numel() for p in model.policy.parameters())
    print(f"  Model parameters: {param_count:,}")
    
    # ─── Train ───
    print(f"\n[4/4] Starting training for {cfg['total_timesteps']:,} timesteps...")
    print("      TensorBoard: tensorboard --logdir guidance_tensorboard/")
    print("-" * 70)
    
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
    )
    
    # ─── Save final model ───
    final_path = os.path.join(cfg["model_dir"], cfg["model_name"])
    model.save(final_path)
    print("\n" + "=" * 70)
    print(f"  Training complete! Model saved to: {final_path}")
    print("=" * 70)
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model


def evaluate_agent(model_path: str, n_episodes: int = 20):
    """
    Evaluate a trained agent and print statistics.
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of evaluation episodes
    """
    print(f"\nEvaluating model: {model_path}")
    
    env = GuidanceEnv()
    model = RecurrentPPO.load(model_path)
    
    hits = 0
    miss_distances = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        done = False
        
        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start,
                deterministic=True
            )
            episode_start = np.zeros((1,), dtype=bool)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        stats = info.get("episode_stats", {})
        if stats.get("hit", False):
            hits += 1
        miss_distances.append(stats.get("min_distance", float('inf')))
    
    hit_rate = hits / n_episodes * 100
    avg_miss = np.mean(miss_distances)
    
    print(f"  Episodes:      {n_episodes}")
    print(f"  Hit rate:      {hit_rate:.1f}%")
    print(f"  Avg miss dist: {avg_miss:.2f} m")
    print(f"  Min miss dist: {min(miss_distances):.2f} m")
    print(f"  Max miss dist: {max(miss_distances):.2f} m")
    
    return {"hit_rate": hit_rate, "avg_miss": avg_miss}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    import argparse
    parser = argparse.ArgumentParser(description="INTERCEPTOR LSTM-PPO Training")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--model-path", type=str, default="models/interceptor_lstm_guidance")
    args = parser.parse_args()
    
    if args.mode == "train":
        config_override = {"total_timesteps": args.timesteps}
        train_guidance_agent(config_override)
    elif args.mode == "eval":
        evaluate_agent(args.model_path)
