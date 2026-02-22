"""
Project INTERCEPTOR — LangGraph Mission Supervisor (VLM Commander)
================================================================

Agentic ground control supervisor built on LangGraph. This state
machine monitors the swarm engagement and makes high-level decisions:

    assess_threat → assign_interceptors → monitor_engagement
          ↑                                       |
          └──── handle_failure ←──────────────────┘

Nodes:
    - assess_threat:       Classify incoming targets, check causal filter
    - assign_interceptors: Allocate interceptors to confirmed targets
    - monitor_engagement:  Track engagement progress, check fuel/status
    - handle_failure:      Reassign interceptors on failure, abort if needed

Conditional Edges:
    - If interceptor fuel < 20% → handle_failure
    - If target classified as decoy → reassess threat
    - If all interceptors lost → mission abort
    - If hit confirmed → complete
"""

import sys
import time
import logging
from pathlib import Path
from typing import Literal

from langgraph.graph import StateGraph, END

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from supervisor.graph_state import MissionState, create_initial_mission_state

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  GRAPH NODES — Each node takes MissionState, returns updates
# ═══════════════════════════════════════════════════════════════

def assess_threat(state: MissionState) -> dict:
    """
    Node 1: Assess and classify incoming threats.
    
    In a full implementation, this would query the Causal AI filter
    and optionally a VLM for visual classification.
    """
    updated_targets = []
    alerts = list(state.get("alerts", []))
    
    for target in state["targets"]:
        target = dict(target)  # Copy
        
        # Check if causal filter flagged this as a decoy
        if target["decoy_confidence"] > 0.7:
            target["classification"] = "suspected_decoy"
            alerts.append({
                "type": "DECOY_ALERT",
                "message": f"Target {target['target_id']} flagged as probable decoy "
                           f"(confidence: {target['decoy_confidence']:.0%})",
                "timestamp": time.time(),
            })
        elif target["decoy_confidence"] < 0.3:
            target["classification"] = "confirmed_real"
        else:
            target["classification"] = "unknown"
        
        updated_targets.append(target)
    
    reasoning = (
        f"Assessed {len(updated_targets)} target(s). "
        f"Real: {sum(1 for t in updated_targets if t['classification'] == 'confirmed_real')}, "
        f"Decoy: {sum(1 for t in updated_targets if t['classification'] == 'suspected_decoy')}, "
        f"Unknown: {sum(1 for t in updated_targets if t['classification'] == 'unknown')}."
    )
    
    logger.info(f"[ASSESS_THREAT] {reasoning}")
    
    return {
        "targets": updated_targets,
        "alerts": alerts,
        "supervisor_action": "THREAT_ASSESSED",
        "supervisor_reasoning": reasoning,
        "phase": "midcourse" if state["phase"] == "pre_launch" else state["phase"],
    }


def assign_interceptors(state: MissionState) -> dict:
    """
    Node 2: Assign interceptors to confirmed real targets.
    
    Simple round-robin assignment for now. In a full MAPPO system,
    this would emit geometry recalculation commands to the RL agents.
    """
    real_targets = [
        t for t in state["targets"]
        if t["classification"] != "suspected_decoy"
    ]
    
    active_interceptors = [
        i for i in state["interceptors"]
        if i["status"] == "active"
    ]
    
    decision_log = list(state.get("decision_log", []))
    
    if not real_targets:
        reasoning = "No confirmed real targets. Holding interceptors."
        decision_log.append({
            "action": "HOLD",
            "reasoning": reasoning,
            "timestamp": time.time()
        })
        return {
            "supervisor_action": "HOLDING",
            "supervisor_reasoning": reasoning,
            "decision_log": decision_log,
        }
    
    # Assign interceptors evenly across real targets
    assignments = {}
    for i, interceptor in enumerate(active_interceptors):
        target_idx = i % len(real_targets)
        target_id = real_targets[target_idx]["target_id"]
        assignments[interceptor["interceptor_id"]] = target_id
    
    reasoning = (
        f"Assigned {len(active_interceptors)} interceptor(s) to "
        f"{len(real_targets)} target(s). "
        f"Assignments: {assignments}"
    )
    
    decision_log.append({
        "action": "ASSIGN",
        "assignments": assignments,
        "reasoning": reasoning,
        "timestamp": time.time()
    })
    
    logger.info(f"[ASSIGN_INTERCEPTORS] {reasoning}")
    
    return {
        "supervisor_action": "INTERCEPTORS_ASSIGNED",
        "supervisor_reasoning": reasoning,
        "reassignment_needed": False,
        "decision_log": decision_log,
        "phase": "terminal",
    }


def monitor_engagement(state: MissionState) -> dict:
    """
    Node 3: Monitor ongoing engagement and check for anomalies.
    
    Checks:
    - Fuel levels across all interceptors
    - Distance/ZEM convergence  
    - Hit/miss detection
    - Diverging interceptors
    """
    alerts = list(state.get("alerts", []))
    updated_interceptors = []
    reassignment_needed = False
    mission_abort = False
    hit_detected = False
    
    for interceptor in state["interceptors"]:
        interceptor = dict(interceptor)
        
        # Check fuel
        if interceptor["fuel_remaining"] < 20.0 and interceptor["status"] == "active":
            interceptor["status"] = "fuel_critical"
            alerts.append({
                "type": "FUEL_CRITICAL",
                "message": f"Interceptor {interceptor['interceptor_id']} fuel critical "
                           f"({interceptor['fuel_remaining']:.1f}%)",
                "timestamp": time.time(),
            })
            reassignment_needed = True
        
        # Check if hit
        if interceptor["distance_to_target"] < 5.0:
            interceptor["status"] = "hit"
            hit_detected = True
            alerts.append({
                "type": "INTERCEPT_SUCCESS",
                "message": f"Interceptor {interceptor['interceptor_id']} HIT! "
                           f"Miss distance: {interceptor['distance_to_target']:.2f}m",
                "timestamp": time.time(),
            })
        
        # Check if diverging (missed)
        if interceptor["closing_velocity"] < -200.0 and interceptor["status"] == "active":
            interceptor["status"] = "missed"
            reassignment_needed = True
        
        updated_interceptors.append(interceptor)
    
    # Count active interceptors
    active_count = sum(1 for i in updated_interceptors if i["status"] in ("active", "fuel_critical"))
    
    # Check for total loss
    if active_count == 0 and not hit_detected:
        mission_abort = True
        alerts.append({
            "type": "MISSION_ABORT",
            "message": "All interceptors lost. Mission abort.",
            "timestamp": time.time(),
        })
    
    phase = state["phase"]
    if hit_detected:
        phase = "complete"
    
    reasoning = (
        f"Monitoring: {active_count}/{state['total_interceptor_count']} active. "
        f"{'HIT CONFIRMED! ' if hit_detected else ''}"
        f"{'Reassignment needed. ' if reassignment_needed else ''}"
        f"{'MISSION ABORT. ' if mission_abort else ''}"
    )
    
    logger.info(f"[MONITOR] {reasoning}")
    
    return {
        "interceptors": updated_interceptors,
        "active_interceptor_count": active_count,
        "alerts": alerts,
        "reassignment_needed": reassignment_needed,
        "mission_abort": mission_abort,
        "phase": phase,
        "supervisor_action": "MONITORING",
        "supervisor_reasoning": reasoning,
    }


def handle_failure(state: MissionState) -> dict:
    """
    Node 4: Handle interceptor failures and reassign.
    
    When an interceptor runs out of fuel or misses, the remaining
    active interceptors must recalculate their interception geometry.
    """
    decision_log = list(state.get("decision_log", []))
    alerts = list(state.get("alerts", []))
    
    failed = [
        i for i in state["interceptors"]
        if i["status"] in ("fuel_critical", "missed", "lost")
    ]
    
    active = [
        i for i in state["interceptors"]
        if i["status"] == "active"
    ]
    
    reasoning = (
        f"Handling failure: {len(failed)} interceptor(s) compromised. "
        f"{len(active)} remaining active. "
    )
    
    if not active:
        reasoning += "No active interceptors remaining — recommending abort."
        decision_log.append({
            "action": "ABORT_RECOMMENDED",
            "reasoning": reasoning,
            "timestamp": time.time(),
        })
    else:
        reasoning += f"Recalculating geometry for interceptors {[i['interceptor_id'] for i in active]}."
        decision_log.append({
            "action": "REASSIGN",
            "active_ids": [i["interceptor_id"] for i in active],
            "reasoning": reasoning,
            "timestamp": time.time(),
        })
        alerts.append({
            "type": "GEOMETRY_RECALC",
            "message": f"Swarm geometry recalculated with {len(active)} interceptors",
            "timestamp": time.time(),
        })
    
    logger.info(f"[HANDLE_FAILURE] {reasoning}")
    
    return {
        "supervisor_action": "FAILURE_HANDLED",
        "supervisor_reasoning": reasoning,
        "reassignment_needed": False,
        "decision_log": decision_log,
        "alerts": alerts,
    }


# ═══════════════════════════════════════════════════════════════
#  CONDITIONAL EDGES — Routing logic between nodes
# ═══════════════════════════════════════════════════════════════

def route_after_monitor(state: MissionState) -> Literal["handle_failure", "assess_threat", "__end__"]:
    """Route after monitoring step."""
    if state.get("mission_abort", False):
        return END
    if state.get("phase") == "complete":
        return END
    if state.get("reassignment_needed", False):
        return "handle_failure"
    return "assess_threat"


def route_after_failure(state: MissionState) -> Literal["assign_interceptors", "__end__"]:
    """Route after handling failure."""
    active = sum(1 for i in state["interceptors"] if i["status"] == "active")
    if active == 0:
        return END
    return "assign_interceptors"


# ═══════════════════════════════════════════════════════════════
#  GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════

def build_mission_graph() -> StateGraph:
    """
    Build and compile the LangGraph mission supervision graph.
    
    Flow:
        assess_threat → assign_interceptors → monitor_engagement
              ↑                                       |
              └──── handle_failure ←──────────────────┘
    
    Returns:
        Compiled LangGraph StateGraph
    """
    graph = StateGraph(MissionState)
    
    # Add nodes
    graph.add_node("assess_threat", assess_threat)
    graph.add_node("assign_interceptors", assign_interceptors)
    graph.add_node("monitor_engagement", monitor_engagement)
    graph.add_node("handle_failure", handle_failure)
    
    # Set entry point
    graph.set_entry_point("assess_threat")
    
    # Add edges
    graph.add_edge("assess_threat", "assign_interceptors")
    graph.add_edge("assign_interceptors", "monitor_engagement")
    
    # Conditional edges after monitoring
    graph.add_conditional_edges(
        "monitor_engagement",
        route_after_monitor,
        {
            "handle_failure": "handle_failure",
            "assess_threat": "assess_threat",
            END: END,
        }
    )
    
    # Conditional edges after failure handling
    graph.add_conditional_edges(
        "handle_failure",
        route_after_failure,
        {
            "assign_interceptors": "assign_interceptors",
            END: END,
        }
    )
    
    compiled = graph.compile()
    return compiled


# ─── Standalone test ───
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    print("=" * 60)
    print("  PROJECT INTERCEPTOR — Mission Supervisor Graph Test")
    print("=" * 60)
    
    # Build the graph
    graph = build_mission_graph()
    print(f"\n✅ Graph compiled successfully: {graph}")
    
    # Create initial state with a scenario that terminates
    state = create_initial_mission_state(
        mission_id="TEST-001",
        n_interceptors=3,
        n_targets=1
    )
    
    # Simulate a scenario: interceptor 0 hits
    state["interceptors"][0]["distance_to_target"] = 3.0  # Below kill radius
    state["interceptors"][0]["closing_velocity"] = 1300.0
    state["targets"][0]["decoy_confidence"] = 0.1  # Confirmed real
    
    print("\n─── Running graph with hit scenario ───")
    
    # Run the graph
    final_state = None
    for step_output in graph.stream(state):
        node_name = list(step_output.keys())[0]
        node_result = step_output[node_name]
        print(f"\n  Node: {node_name}")
        print(f"  Action: {node_result.get('supervisor_action', 'N/A')}")
        print(f"  Reasoning: {node_result.get('supervisor_reasoning', 'N/A')}")
        final_state = node_result
    
    print("\n" + "=" * 60)
    print(f"  Mission phase: {final_state.get('phase', 'unknown')}")
    print(f"  Alerts: {len(final_state.get('alerts', []))}")
    print("=" * 60)
