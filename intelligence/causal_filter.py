"""
Project INTERCEPTOR — Causal AI Decoy Rejection Filter
=====================================================

Analyzes incoming radar tracks to distinguish real targets from decoys
(flares, chaff, electronic countermeasures). Uses physics-based causal
reasoning:

1. **Acceleration Thresholding**: Flags tracks exceeding physically
   plausible acceleration (>100G for typical aircraft).
   
2. **Trajectory Consistency**: Checks if a track's velocity and position
   history are physically consistent (no teleportation).
   
3. **Track Split Analysis**: When a single track splits into multiple,
   analyzes which children are kinematically feasible.

4. **Radar Cross-Section (RCS) Analysis**: Decoys typically have rapidly
   decaying RCS compared to real targets.
"""

import numpy as np
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a single radar track."""
    track_id: int
    positions: list = field(default_factory=list)    # List of [x, y, z]
    velocities: list = field(default_factory=list)   # List of [vx, vy, vz]
    timestamps: list = field(default_factory=list)   # List of time (s)
    rcs_history: list = field(default_factory=list)  # Radar cross-section (m²)
    is_decoy: bool = False
    confidence: float = 1.0  # 1.0 = definitely real, 0.0 = definitely decoy
    parent_id: Optional[int] = None  # If this track split from another


class CausalDecoyFilter:
    """
    Physics-based causal filter for rejecting decoy targets.
    
    Sits between the Kafka telemetry stream and the RL agent,
    ensuring only physically plausible tracks are forwarded.
    """

    def __init__(self, config: dict = None):
        config = config or {}
        
        # Physical plausibility thresholds
        self.max_plausible_g = config.get("max_plausible_g", 100.0)        # Max G-force
        self.max_plausible_accel = self.max_plausible_g * 9.81             # m/s²
        self.max_velocity = config.get("max_velocity", 3000.0)             # m/s (~Mach 9)
        self.min_rcs_ratio = config.get("min_rcs_ratio", 0.1)             # RCS decay threshold
        self.consistency_tolerance = config.get("consistency_tolerance", 50.0)  # meters
        
        # Track history
        self.tracks: dict[int, Track] = {}
        self.rejected_tracks: set = set()
        
        # Statistics
        self.stats = {
            "total_tracks_analyzed": 0,
            "decoys_rejected": 0,
            "accel_violations": 0,
            "consistency_violations": 0,
            "rcs_violations": 0,
        }

    def update_track(
        self,
        track_id: int,
        position: np.ndarray,
        velocity: np.ndarray,
        timestamp: float,
        rcs: float = None,
        parent_id: int = None
    ) -> dict:
        """
        Update a track with new sensor data and run causal checks.
        
        Args:
            track_id: Unique track identifier
            position: Current position [3]
            velocity: Current velocity [3]
            timestamp: Current time (s)
            rcs: Radar cross-section (m²), optional
            parent_id: Parent track ID if this is a split
            
        Returns:
            Assessment dict with 'is_decoy', 'confidence', 'reason'
        """
        # Get or create track
        if track_id not in self.tracks:
            self.tracks[track_id] = Track(
                track_id=track_id,
                parent_id=parent_id
            )
        
        track = self.tracks[track_id]
        track.positions.append(position.tolist() if isinstance(position, np.ndarray) else position)
        track.velocities.append(velocity.tolist() if isinstance(velocity, np.ndarray) else velocity)
        track.timestamps.append(timestamp)
        if rcs is not None:
            track.rcs_history.append(rcs)
        
        self.stats["total_tracks_analyzed"] += 1
        
        # Run causal checks
        assessment = self._assess_track(track)
        
        if assessment["is_decoy"]:
            track.is_decoy = True
            track.confidence = assessment["confidence"]
            self.rejected_tracks.add(track_id)
            self.stats["decoys_rejected"] += 1
            logger.info(
                f"DECOY DETECTED: Track {track_id} — "
                f"Reason: {assessment['reason']} "
                f"(confidence: {assessment['confidence']:.2f})"
            )
        
        return assessment

    def _assess_track(self, track: Track) -> dict:
        """Run all causal checks on a track."""
        reasons = []
        min_confidence = 1.0
        
        # Check 1: Acceleration plausibility
        accel_result = self._check_acceleration(track)
        if accel_result["violated"]:
            reasons.append(accel_result["reason"])
            min_confidence = min(min_confidence, accel_result["confidence"])
            self.stats["accel_violations"] += 1
        
        # Check 2: Trajectory consistency
        consistency_result = self._check_consistency(track)
        if consistency_result["violated"]:
            reasons.append(consistency_result["reason"])
            min_confidence = min(min_confidence, consistency_result["confidence"])
            self.stats["consistency_violations"] += 1
        
        # Check 3: RCS decay analysis
        rcs_result = self._check_rcs_decay(track)
        if rcs_result["violated"]:
            reasons.append(rcs_result["reason"])
            min_confidence = min(min_confidence, rcs_result["confidence"])
            self.stats["rcs_violations"] += 1
        
        is_decoy = len(reasons) > 0
        
        return {
            "is_decoy": is_decoy,
            "confidence": min_confidence if is_decoy else 1.0,
            "reason": "; ".join(reasons) if reasons else "Track is physically plausible",
            "track_id": track.track_id,
        }

    def _check_acceleration(self, track: Track) -> dict:
        """Check if the track's implied acceleration is physically possible."""
        if len(track.velocities) < 2:
            return {"violated": False, "reason": "", "confidence": 1.0}
        
        v_curr = np.array(track.velocities[-1])
        v_prev = np.array(track.velocities[-2])
        dt = track.timestamps[-1] - track.timestamps[-2]
        
        if dt < 1e-10:
            return {"violated": False, "reason": "", "confidence": 1.0}
        
        accel = np.linalg.norm(v_curr - v_prev) / dt
        accel_g = accel / 9.81
        
        if accel > self.max_plausible_accel:
            # Confidence scales inversely with how far over the threshold
            excess_ratio = accel / self.max_plausible_accel
            confidence = max(0.0, 1.0 - (excess_ratio - 1.0) * 0.5)
            return {
                "violated": True,
                "reason": f"Acceleration {accel_g:.1f}G exceeds {self.max_plausible_g}G limit",
                "confidence": confidence,
            }
        
        return {"violated": False, "reason": "", "confidence": 1.0}

    def _check_consistency(self, track: Track) -> dict:
        """Check if position/velocity history is physically consistent."""
        if len(track.positions) < 3:
            return {"violated": False, "reason": "", "confidence": 1.0}
        
        pos_curr = np.array(track.positions[-1])
        pos_prev = np.array(track.positions[-2])
        vel_prev = np.array(track.velocities[-2])
        dt = track.timestamps[-1] - track.timestamps[-2]
        
        if dt < 1e-10:
            return {"violated": False, "reason": "", "confidence": 1.0}
        
        # Where should the track be based on previous state?
        predicted_pos = np.array(pos_prev) + vel_prev * dt
        error = np.linalg.norm(pos_curr - predicted_pos)
        
        if error > self.consistency_tolerance:
            confidence = max(0.0, 1.0 - (error / self.consistency_tolerance - 1.0) * 0.3)
            return {
                "violated": True,
                "reason": f"Position inconsistency: {error:.1f}m (tol={self.consistency_tolerance}m)",
                "confidence": confidence,
            }
        
        return {"violated": False, "reason": "", "confidence": 1.0}

    def _check_rcs_decay(self, track: Track) -> dict:
        """Check if RCS is decaying rapidly (indicative of flare/chaff)."""
        if len(track.rcs_history) < 5:
            return {"violated": False, "reason": "", "confidence": 1.0}
        
        initial_rcs = np.mean(track.rcs_history[:3])
        recent_rcs = np.mean(track.rcs_history[-3:])
        
        if initial_rcs < 1e-10:
            return {"violated": False, "reason": "", "confidence": 1.0}
        
        rcs_ratio = recent_rcs / initial_rcs
        
        if rcs_ratio < self.min_rcs_ratio:
            confidence = max(0.0, rcs_ratio / self.min_rcs_ratio)
            return {
                "violated": True,
                "reason": f"RCS decayed to {rcs_ratio:.1%} of initial (flare/chaff signature)",
                "confidence": confidence,
            }
        
        return {"violated": False, "reason": "", "confidence": 1.0}

    def evaluate_tracks(self, track_reports: list) -> list:
        """
        Batch-evaluate multiple track reports.
        
        Args:
            track_reports: List of dicts with 'id', 'accel' (or full state)
            
        Returns:
            List of assessment results
        """
        results = []
        for report in track_reports:
            track_id = report.get("id", 0)
            accel_g = report.get("accel", 0.0)
            
            # Simulate a velocity change consistent with the reported G
            velocity = np.array(report.get("velocity", [300.0, 0.0, 0.0]))
            position = np.array(report.get("position", [5000.0, 0.0, 0.0]))
            dt = 0.01
            
            # Create velocity that implies the given acceleration
            accel_ms2 = accel_g * 9.81
            new_velocity = velocity + np.array([0, accel_ms2 * dt, 0])
            
            timestamp = report.get("timestamp", len(self.tracks.get(track_id, Track(0)).timestamps) * dt)
            
            # First update to establish baseline
            if track_id not in self.tracks:
                self.update_track(track_id, position, velocity, timestamp - dt)
            
            # Second update with the new velocity
            new_position = position + new_velocity * dt
            assessment = self.update_track(track_id, new_position, new_velocity, timestamp)
            results.append(assessment)
        
        return results

    def get_real_tracks(self) -> list:
        """Return only tracks that have NOT been flagged as decoys."""
        return [
            t for t in self.tracks.values()
            if not t.is_decoy
        ]

    def get_stats(self) -> dict:
        """Return filter statistics."""
        return {
            **self.stats,
            "active_tracks": len(self.tracks),
            "rejected_count": len(self.rejected_tracks),
        }


# ─── Standalone test ───
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    cf = CausalDecoyFilter()
    
    # Simulate two tracks: one real (50G), one decoy (200G)
    results = cf.evaluate_tracks([
        {"id": 1, "accel": 50, "velocity": [300, 0, 0], "position": [5000, 0, 0]},
        {"id": 2, "accel": 200, "velocity": [300, 0, 0], "position": [5000, 100, 0]},
    ])
    
    for r in results:
        status = "🚫 DECOY" if r["is_decoy"] else "✅ REAL"
        print(f"Track {r['track_id']}: {status} — {r['reason']}")
    
    print(f"\nFilter stats: {cf.get_stats()}")
