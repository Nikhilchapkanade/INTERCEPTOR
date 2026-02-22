"""
Project INTERCEPTOR — Kafka Telemetry Producer
=============================================

Streams simulation state to Apache Kafka topics at each simulation step.
Publishes engagement telemetry (positions, velocities, ZEM, accelerations)
as JSON payloads to the 'missile_telemetry' topic.

Also publishes engagement events (hit, miss, interceptor-lost) to
the 'engagement_events' topic for the LangGraph supervisor.
"""

import json
import time
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


class TelemetryStreamer:
    """
    Streams real-time simulation telemetry to Kafka.
    
    Falls back to in-memory logging if Kafka is unavailable,
    allowing the system to run without a Kafka cluster.
    """

    def __init__(
        self,
        telemetry_topic: str = 'missile_telemetry',
        events_topic: str = 'engagement_events',
        bootstrap_servers: list = None,
        enable_kafka: bool = True
    ):
        self.telemetry_topic = telemetry_topic
        self.events_topic = events_topic
        self.bootstrap_servers = bootstrap_servers or ['localhost:9092']
        self.producer = None
        self.kafka_available = False
        self.message_buffer = []  # In-memory fallback buffer
        self.messages_sent = 0
        
        if enable_kafka:
            self._connect_kafka()
        else:
            logger.info("Kafka disabled. Using in-memory telemetry buffer.")

    def _connect_kafka(self):
        """Attempt to connect to Kafka cluster."""
        try:
            from kafka import KafkaProducer
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, cls=NumpyEncoder).encode('utf-8'),
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1,
            )
            self.kafka_available = True
            logger.info(f"Connected to Kafka at {self.bootstrap_servers}")
        except Exception as e:
            logger.warning(f"Kafka unavailable ({e}). Falling back to in-memory buffer.")
            self.kafka_available = False

    def stream_state(self, raw_state: dict, interceptor_id: int = 0):
        """
        Stream a single simulation state snapshot.

        Args:
            raw_state: Dict from GuidanceEnv.get_raw_state()
            interceptor_id: ID of the interceptor (for multi-agent)
        """
        payload = {
            "interceptor_id": interceptor_id,
            "timestamp_ms": int(time.time() * 1000),
            **raw_state
        }
        
        if self.kafka_available:
            try:
                self.producer.send(self.telemetry_topic, payload)
                self.messages_sent += 1
                # Flush periodically (every 100 messages)
                if self.messages_sent % 100 == 0:
                    self.producer.flush()
            except Exception as e:
                logger.error(f"Kafka send failed: {e}")
                self.message_buffer.append(payload)
        else:
            self.message_buffer.append(payload)
            # Keep buffer bounded
            if len(self.message_buffer) > 10000:
                self.message_buffer = self.message_buffer[-5000:]

    def stream_event(self, event_type: str, details: dict, interceptor_id: int = 0):
        """
        Stream an engagement event (hit, miss, fuel_exhausted, etc.)

        Args:
            event_type: Event classification string
            details: Additional event context
            interceptor_id: ID of the interceptor
        """
        payload = {
            "interceptor_id": interceptor_id,
            "event_type": event_type,
            "timestamp_ms": int(time.time() * 1000),
            "details": details
        }
        
        if self.kafka_available:
            try:
                self.producer.send(self.events_topic, payload)
                self.producer.flush()
            except Exception as e:
                logger.error(f"Kafka event send failed: {e}")
                self.message_buffer.append(payload)
        else:
            self.message_buffer.append(payload)
            logger.info(f"EVENT [{event_type}] Interceptor {interceptor_id}: {details}")

    def get_buffer(self) -> list:
        """Return the in-memory message buffer (for non-Kafka mode)."""
        return self.message_buffer.copy()

    def flush(self):
        """Force-flush all pending Kafka messages."""
        if self.kafka_available and self.producer:
            self.producer.flush()

    def close(self):
        """Cleanly shut down the producer."""
        if self.kafka_available and self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed.")


# ─── Standalone test ───
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    streamer = TelemetryStreamer(enable_kafka=False)
    
    # Simulate streaming 10 mock states
    for i in range(10):
        mock_state = {
            "time_step": i,
            "time_seconds": i * 0.01,
            "interceptor_pos": [i * 100.0, 0.0, 0.0],
            "target_pos": [10000.0 - i * 30.0, 50.0, -20.0],
            "distance": 10000.0 - i * 130.0,
            "zem_norm": 45.0 - i * 3.0,
            "closing_velocity": 1300.0,
        }
        streamer.stream_state(mock_state, interceptor_id=0)
    
    streamer.stream_event("INTERCEPT_HIT", {"miss_distance": 3.2}, interceptor_id=0)
    
    print(f"Buffer contains {len(streamer.get_buffer())} messages")
    print(f"Last event: {streamer.get_buffer()[-1]}")
