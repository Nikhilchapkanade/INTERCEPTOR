"""
Project INTERCEPTOR — Missile Guidance Gymnasium Environment
==========================================================

Custom Gymnasium environment for training RL agents on 3-DOF missile
guidance against maneuvering targets. The agent controls lateral
acceleration commands (pitch and yaw) while the simulation propagates
both interceptor and target kinematics at 100Hz.

State Space (12-dim):
    - Relative position [3]: (target_pos - interceptor_pos)
    - Relative velocity [3]: (target_vel - interceptor_vel)
    - Zero-Effort-Miss [3]:  ZEM vector
    - Target acceleration [3]: estimated target maneuver

Action Space (2-dim):
    - Commanded lateral acceleration [a_y, a_z] bounded ±30G

Reward:
    R = -k₁ · ‖ZEM‖ - k₂ · ‖aᶜ‖² + R_hit
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulation.kinematics import compute_zem, compute_t_go, compute_closing_velocity


class GuidanceEnv(gym.Env):
    """
    3-DOF Missile Guidance Environment.
    
    The interceptor starts at the origin traveling along +X at ~Mach 3.
    The target is initialized downrange at ~10km with random lateral offset
    and an incoming velocity (head-on engagement).
    
    The target performs evasive sinusoidal maneuvers in Y and Z.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config: dict = None):
        super(GuidanceEnv, self).__init__()
        
        config = config or {}
        
        # --- Configurable parameters ---
        self.max_g = config.get("max_g", 30.0) * 9.81          # Max accel (m/s²)
        self.dt = config.get("dt", 0.01)                        # Timestep (100Hz)
        self.max_steps = config.get("max_steps", 1500)           # Episode length limit
        self.hit_radius = config.get("hit_radius", 5.0)         # Kill radius (m)
        self.initial_range = config.get("initial_range", 10000.0)
        self.interceptor_speed = config.get("interceptor_speed", 1000.0)  # ~Mach 3
        self.target_speed = config.get("target_speed", 300.0)
        self.target_maneuver_g = config.get("target_maneuver_g", 7.0)    # Target max G
        
        # Reward coefficients
        self.k_zem = config.get("k_zem", 0.001)       # ZEM penalty weight
        self.k_accel = config.get("k_accel", 1e-6)    # Control effort penalty
        self.r_hit = config.get("r_hit", 500.0)        # Hit bonus
        self.r_miss = config.get("r_miss", -100.0)     # Miss penalty
        
        # --- Spaces ---
        # Action: lateral acceleration commands [a_y, a_z]
        self.action_space = spaces.Box(
            low=-self.max_g, 
            high=self.max_g, 
            shape=(2,), 
            dtype=np.float32
        )
        
        # Observation: [rel_pos(3), rel_vel(3), zem(3), target_accel_est(3)]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(12,), 
            dtype=np.float32
        )
        
        # --- Internal state ---
        self.interceptor_pos = np.zeros(3)
        self.interceptor_vel = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.target_vel = np.zeros(3)
        self.target_accel = np.zeros(3)
        self.time_step = 0
        self.prev_distance = 0.0
        
        # Episode statistics
        self.episode_stats = {
            "min_distance": np.inf,
            "total_accel_used": 0.0,
            "hit": False,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Interceptor: origin, heading +X at Mach 3
        self.interceptor_pos = np.array([0.0, 0.0, 0.0])
        self.interceptor_vel = np.array([self.interceptor_speed, 0.0, 0.0])
        
        # Target: downrange with random lateral offset
        lateral_spread = self.initial_range * 0.05  # 5% of range
        self.target_pos = np.array([
            self.initial_range,
            self.np_random.uniform(-lateral_spread, lateral_spread),
            self.np_random.uniform(-lateral_spread, lateral_spread)
        ])
        
        # Target incoming velocity with slight randomization
        speed_var = self.np_random.uniform(0.8, 1.2)
        angle_var = self.np_random.uniform(-0.1, 0.1)
        self.target_vel = np.array([
            -self.target_speed * speed_var,
            self.target_speed * angle_var,
            self.target_speed * angle_var * 0.5
        ])
        
        self.target_accel = np.zeros(3)
        self.time_step = 0
        self.prev_distance = np.linalg.norm(self.target_pos - self.interceptor_pos)
        
        # Reset episode stats
        self.episode_stats = {
            "min_distance": self.prev_distance,
            "total_accel_used": 0.0,
            "hit": False,
        }
        
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -self.max_g, self.max_g)
        
        # ─── 1. Target evasive maneuver ───
        # Sinusoidal weave with frequency that increases over time (harder to predict)
        t = self.time_step * self.dt
        maneuver_mag = self.target_maneuver_g * 9.81
        freq_y = 0.5 + 0.3 * np.sin(0.1 * t)  # Time-varying frequency
        freq_z = 0.7 + 0.2 * np.cos(0.15 * t)
        
        self.target_accel = np.array([
            0.0,
            maneuver_mag * np.sin(2 * np.pi * freq_y * t),
            maneuver_mag * np.cos(2 * np.pi * freq_z * t)
        ])
        
        # Add random jerk for unpredictability
        jerk = self.np_random.normal(0, maneuver_mag * 0.1, size=3)
        jerk[0] = 0.0  # No longitudinal jerk
        self.target_accel += jerk
        
        # Propagate target state
        self.target_vel += self.target_accel * self.dt
        self.target_pos += self.target_vel * self.dt
        
        # ─── 2. Apply interceptor control ───
        interceptor_accel = np.array([0.0, action[0], action[1]])
        self.interceptor_vel += interceptor_accel * self.dt
        self.interceptor_pos += self.interceptor_vel * self.dt
        
        # ─── 3. Compute engagement geometry ───
        rel_pos = self.target_pos - self.interceptor_pos
        rel_vel = self.target_vel - self.interceptor_vel
        distance = np.linalg.norm(rel_pos)
        zem = compute_zem(rel_pos, rel_vel)
        zem_norm = np.linalg.norm(zem)
        t_go = compute_t_go(rel_pos, rel_vel)
        v_c = compute_closing_velocity(rel_pos, rel_vel)
        
        # Update stats
        self.episode_stats["min_distance"] = min(self.episode_stats["min_distance"], distance)
        self.episode_stats["total_accel_used"] += np.linalg.norm(action)
        
        # ─── 4. Reward computation ───
        # Core: penalize ZEM magnitude and control effort
        reward = -self.k_zem * zem_norm - self.k_accel * np.sum(np.square(action))
        
        # Shaping: reward for decreasing distance (approach reward)
        distance_delta = self.prev_distance - distance
        reward += 0.01 * distance_delta
        
        # Bonus for keeping ZEM small as we get close
        if t_go < 2.0 and zem_norm < 50.0:
            reward += 0.5
        
        # ─── 5. Termination ───
        terminated = False
        truncated = False
        
        # Hit condition
        if distance < self.hit_radius:
            terminated = True
            reward += self.r_hit
            self.episode_stats["hit"] = True
        
        # Miss conditions
        if self.time_step >= self.max_steps:
            truncated = True
            reward += self.r_miss
        
        # Diverging (interceptor flew past target)
        if v_c < -100.0 and distance > 500.0:
            truncated = True
            reward += self.r_miss
        
        self.prev_distance = distance
        self.time_step += 1
        
        info = {
            "distance": distance,
            "zem_norm": zem_norm,
            "t_go": t_go,
            "closing_velocity": v_c,
            "episode_stats": self.episode_stats if (terminated or truncated) else {},
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Construct the 12-dimensional observation vector."""
        rel_pos = self.target_pos - self.interceptor_pos
        rel_vel = self.target_vel - self.interceptor_vel
        zem = compute_zem(rel_pos, rel_vel)
        
        # In a real system, target acceleration would be estimated from tracking
        # Here we add noise to simulate imperfect estimation
        accel_noise = np.random.normal(0, 5.0, size=3)
        target_accel_est = self.target_accel + accel_noise
        
        obs = np.concatenate([
            rel_pos / self.initial_range,          # Normalize by initial range
            rel_vel / self.interceptor_speed,      # Normalize by interceptor speed
            zem / self.initial_range,              # Normalize ZEM
            target_accel_est / (self.max_g),       # Normalize by max accel
        ]).astype(np.float32)
        
        return obs

    def get_raw_state(self) -> dict:
        """Return the raw (un-normalized) state for telemetry streaming."""
        rel_pos = self.target_pos - self.interceptor_pos
        rel_vel = self.target_vel - self.interceptor_vel
        zem = compute_zem(rel_pos, rel_vel)
        
        return {
            "time_step": self.time_step,
            "time_seconds": self.time_step * self.dt,
            "interceptor_pos": self.interceptor_pos.tolist(),
            "interceptor_vel": self.interceptor_vel.tolist(),
            "target_pos": self.target_pos.tolist(),
            "target_vel": self.target_vel.tolist(),
            "target_accel": self.target_accel.tolist(),
            "relative_position": rel_pos.tolist(),
            "relative_velocity": rel_vel.tolist(),
            "zem": zem.tolist(),
            "distance": float(np.linalg.norm(rel_pos)),
            "zem_norm": float(np.linalg.norm(zem)),
            "closing_velocity": float(compute_closing_velocity(rel_pos, rel_vel)),
            "t_go": float(compute_t_go(rel_pos, rel_vel)),
        }
