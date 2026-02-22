"""
Kinematics Utilities for Project INTERCEPTOR
==========================================

Mathematical functions for missile guidance computations:
- Zero-Effort-Miss (ZEM) vector
- Line-of-Sight (LOS) rate
- Closing velocity
- Euler angle to Direction Cosine Matrix (DCM) conversion
- Proportional Navigation guidance law (for baseline comparison)
"""

import numpy as np
from scipy.spatial.transform import Rotation


def compute_zem(rel_pos: np.ndarray, rel_vel: np.ndarray) -> np.ndarray:
    """
    Compute the Zero-Effort-Miss (ZEM) vector.
    
    ZEM represents where the interceptor would miss the target if no further
    corrections are applied. It is the key input to proportional navigation.

    ZEM = r + v * t_go
    t_go = -dot(r, v) / dot(v, v)

    Args:
        rel_pos: Relative position vector (target - interceptor) [3]
        rel_vel: Relative velocity vector (target - interceptor) [3]

    Returns:
        ZEM vector [3]
    """
    vel_sq = np.dot(rel_vel, rel_vel)
    if vel_sq < 1e-10:
        return rel_pos.copy()
    
    # Time-to-go estimate (closing time)
    t_go = -np.dot(rel_pos, rel_vel) / vel_sq
    t_go = max(t_go, 1e-6)  # Prevent negative or zero t_go
    
    zem = rel_pos + rel_vel * t_go
    return zem


def compute_t_go(rel_pos: np.ndarray, rel_vel: np.ndarray) -> float:
    """
    Estimate time-to-go until closest approach.

    t_go = -dot(r, v) / dot(v, v)

    Args:
        rel_pos: Relative position vector [3]
        rel_vel: Relative velocity vector [3]

    Returns:
        Estimated time-to-go in seconds
    """
    vel_sq = np.dot(rel_vel, rel_vel)
    if vel_sq < 1e-10:
        return np.linalg.norm(rel_pos) / 1.0  # Fallback
    
    t_go = -np.dot(rel_pos, rel_vel) / vel_sq
    return max(t_go, 1e-6)


def compute_los_rate(rel_pos: np.ndarray, rel_vel: np.ndarray) -> np.ndarray:
    """
    Compute the Line-of-Sight (LOS) angular rate vector.

    omega_LOS = (r × v) / |r|^2

    This is the rate at which the line connecting interceptor to target
    is rotating in inertial space.

    Args:
        rel_pos: Relative position vector [3]
        rel_vel: Relative velocity vector [3]

    Returns:
        LOS rate vector [3] (rad/s)
    """
    r_sq = np.dot(rel_pos, rel_pos)
    if r_sq < 1e-10:
        return np.zeros(3)
    
    omega = np.cross(rel_pos, rel_vel) / r_sq
    return omega


def compute_closing_velocity(rel_pos: np.ndarray, rel_vel: np.ndarray) -> float:
    """
    Compute the scalar closing velocity (approach speed).

    V_c = -dot(r_hat, v_rel)

    Positive values mean the objects are approaching each other.

    Args:
        rel_pos: Relative position vector [3]
        rel_vel: Relative velocity vector [3]

    Returns:
        Closing velocity (m/s), positive = closing
    """
    r_norm = np.linalg.norm(rel_pos)
    if r_norm < 1e-10:
        return 0.0
    
    r_hat = rel_pos / r_norm
    v_c = -np.dot(r_hat, rel_vel)
    return v_c


def euler_to_dcm(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to Direction Cosine Matrix.

    Uses the 3-2-1 (ZYX) rotation sequence standard in aerospace.

    Args:
        phi: Roll angle (rad)
        theta: Pitch angle (rad)
        psi: Yaw angle (rad)

    Returns:
        3x3 Direction Cosine Matrix (body-to-inertial)
    """
    rotation = Rotation.from_euler('ZYX', [psi, theta, phi])
    return rotation.as_matrix()


def proportional_navigation(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    N: float = 4.0
) -> np.ndarray:
    """
    Classical Proportional Navigation (PN) guidance law.
    
    a_c = N * V_c * omega_LOS

    Used as a baseline comparison for the RL agent's performance.

    Args:
        rel_pos: Relative position vector [3]
        rel_vel: Relative velocity vector [3]
        N: Navigation constant (typically 3-5)

    Returns:
        Commanded acceleration vector [3] (m/s^2)
    """
    v_c = compute_closing_velocity(rel_pos, rel_vel)
    omega = compute_los_rate(rel_pos, rel_vel)
    
    a_cmd = N * v_c * omega
    return a_cmd


def augmented_proportional_navigation(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    target_accel: np.ndarray,
    N: float = 4.0
) -> np.ndarray:
    """
    Augmented Proportional Navigation (APN) guidance law.

    a_c = N * V_c * omega_LOS + (N/2) * a_T

    Adds a term to account for known target acceleration,
    reducing the ZEM more aggressively against maneuvering targets.

    Args:
        rel_pos: Relative position vector [3]
        rel_vel: Relative velocity vector [3]
        target_accel: Estimated target acceleration [3]
        N: Navigation constant (typically 3-5)

    Returns:
        Commanded acceleration vector [3] (m/s^2)
    """
    a_pn = proportional_navigation(rel_pos, rel_vel, N)
    a_apn = a_pn + (N / 2.0) * target_accel
    return a_apn


def compute_miss_distance(
    interceptor_pos: np.ndarray,
    interceptor_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    dt: float = 0.001,
    max_time: float = 30.0
) -> float:
    """
    Propagate both bodies ballistically (no control) to find the minimum
    distance of closest approach.

    Used for post-episode analysis and ZEM validation.

    Args:
        interceptor_pos: Interceptor position [3]
        interceptor_vel: Interceptor velocity [3]
        target_pos: Target position [3]
        target_vel: Target velocity [3]
        dt: Integration timestep
        max_time: Maximum propagation time

    Returns:
        Minimum distance between the two bodies (meters)
    """
    min_dist = np.inf
    i_pos = interceptor_pos.copy()
    t_pos = target_pos.copy()
    
    steps = int(max_time / dt)
    for _ in range(steps):
        i_pos += interceptor_vel * dt
        t_pos += target_vel * dt
        dist = np.linalg.norm(t_pos - i_pos)
        
        if dist < min_dist:
            min_dist = dist
        
        # Early exit if diverging
        if dist > min_dist * 2.0 and dist > 100.0:
            break
    
    return min_dist
