"""
Project INTERCEPTOR — LangGraph Mission State Definitions
========================================================

TypedDict-based state definitions for the LangGraph mission
supervisor. Tracks the full engagement state including:
- Interceptor fleet status
- Target classification
- Engagement phase
- Supervisor decisions
"""

from typing import TypedDict, Optional, Literal
from dataclasses import dataclass


class InterceptorStatus(TypedDict):
    """Status of a single interceptor in the swarm."""
    interceptor_id: int
    status: str              # 'active', 'fuel_critical', 'lost', 'hit', 'missed'
    fuel_remaining: float    # Percentage (0-100)
    distance_to_target: float
    zem_norm: float
    closing_velocity: float
    t_go: float
    current_g_load: float


class TargetStatus(TypedDict):
    """Status of the tracked target."""
    target_id: int
    classification: str     # 'confirmed_real', 'suspected_decoy', 'unknown'
    position: list          # [x, y, z]
    velocity: list          # [vx, vy, vz]
    estimated_accel_g: float
    decoy_confidence: float  # 0.0 = definitely real, 1.0 = definitely decoy


class MissionState(TypedDict):
    """
    Full mission state for the LangGraph supervisor graph.
    
    This is the state object that flows through all nodes in
    the mission supervision graph.
    """
    # ─── Engagement status ───
    mission_id: str
    phase: str                          # 'pre_launch', 'boost', 'midcourse', 'terminal', 'complete'
    time_elapsed: float                 # Seconds since engagement start
    
    # ─── Fleet status ───
    interceptors: list                  # List of InterceptorStatus dicts
    active_interceptor_count: int
    total_interceptor_count: int
    
    # ─── Target status ───
    targets: list                       # List of TargetStatus dicts
    primary_target_id: int
    
    # ─── Supervisor decisions ───
    supervisor_action: str              # Latest decision from the supervisor
    supervisor_reasoning: str           # Explanation for the decision
    reassignment_needed: bool           # Flag for interceptor reassignment
    mission_abort: bool                 # Emergency abort flag
    
    # ─── History ───
    decision_log: list                  # List of past supervisor decisions
    alerts: list                        # Active alerts for human operator


def create_initial_mission_state(
    mission_id: str,
    n_interceptors: int = 3,
    n_targets: int = 1
) -> MissionState:
    """
    Create the initial mission state for a new engagement.
    
    Args:
        mission_id: Unique mission identifier
        n_interceptors: Number of interceptors in the swarm
        n_targets: Number of targets to track
        
    Returns:
        Initialized MissionState
    """
    interceptors = [
        InterceptorStatus(
            interceptor_id=i,
            status="active",
            fuel_remaining=100.0,
            distance_to_target=10000.0,
            zem_norm=500.0,
            closing_velocity=1300.0,
            t_go=7.7,
            current_g_load=0.0,
        )
        for i in range(n_interceptors)
    ]
    
    targets = [
        TargetStatus(
            target_id=i,
            classification="unknown",
            position=[10000.0, 0.0, 0.0],
            velocity=[-300.0, 0.0, 0.0],
            estimated_accel_g=0.0,
            decoy_confidence=0.0,
        )
        for i in range(n_targets)
    ]
    
    return MissionState(
        mission_id=mission_id,
        phase="pre_launch",
        time_elapsed=0.0,
        interceptors=interceptors,
        active_interceptor_count=n_interceptors,
        total_interceptor_count=n_interceptors,
        targets=targets,
        primary_target_id=0,
        supervisor_action="INITIALIZING",
        supervisor_reasoning="Mission state initialized. Awaiting engagement.",
        reassignment_needed=False,
        mission_abort=False,
        decision_log=[],
        alerts=[],
    )
