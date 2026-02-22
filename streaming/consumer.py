"""
Project INTERCEPTOR — Kafka Telemetry Consumer
=============================================

Subscribes to Kafka topics and provides the latest telemetry data
to downstream consumers (RL agent, Causal AI filter, Supervisor).

Supports both real-time Kafka consumption and replay from in-memory
buffers for development without a Kafka cluster.
"""

import json
import time
import threading
import logging
from typing import Optional, Callable
from collections import deque

logger = logging.getLogger(__name__)


class TelemetryConsumer:
    """
    Consumes real-time telemetry from Kafka topics.
    
    Runs a background thread that continuously polls Kafka and maintains
    a sliding window of the latest N telemetry snapshots for each
    interceptor.
    """

    def __init__(
        self,
        telemetry_topic: str = 'missile_telemetry',
        events_topic: str = 'engagement_events',
        bootstrap_servers: list = None,
        group_id: str = 'interceptor_consumer_group',
        window_size: int = 100,
        enable_kafka: bool = True
    ):
        self.telemetry_topic = telemetry_topic
        self.events_topic = events_topic
        self.bootstrap_servers = bootstrap_servers or ['localhost:9092']
        self.group_id = group_id
        self.window_size = window_size
        self.kafka_available = False
        
        # Thread-safe storage for latest states per interceptor
        self._lock = threading.Lock()
        self._telemetry_windows: dict[int, deque] = {}  # interceptor_id -> deque
        self._latest_events: list = []
        self._event_callbacks: list[Callable] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        if enable_kafka:
            self._connect_kafka()

    def _connect_kafka(self):
        """Attempt to connect to Kafka cluster."""
        try:
            from kafka import KafkaConsumer
            self.consumer = KafkaConsumer(
                self.telemetry_topic,
                self.events_topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                consumer_timeout_ms=1000,
            )
            self.kafka_available = True
            logger.info(f"Kafka consumer connected to {self.bootstrap_servers}")
        except Exception as e:
            logger.warning(f"Kafka unavailable ({e}). Consumer in offline mode.")
            self.kafka_available = False

    def start(self):
        """Start the background consumption thread."""
        if not self.kafka_available:
            logger.warning("Cannot start consumer — Kafka not available.")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()
        logger.info("Telemetry consumer started.")

    def stop(self):
        """Stop the background consumption thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self.kafka_available:
            self.consumer.close()
        logger.info("Telemetry consumer stopped.")

    def _consume_loop(self):
        """Main consumption loop running in background thread."""
        while self._running:
            try:
                messages = self.consumer.poll(timeout_ms=100)
                for topic_partition, records in messages.items():
                    for record in records:
                        self._process_message(
                            topic_partition.topic,
                            record.value
                        )
            except Exception as e:
                logger.error(f"Consumer error: {e}")
                time.sleep(0.5)

    def _process_message(self, topic: str, payload: dict):
        """Route and store an incoming message."""
        with self._lock:
            if topic == self.telemetry_topic:
                iid = payload.get("interceptor_id", 0)
                if iid not in self._telemetry_windows:
                    self._telemetry_windows[iid] = deque(maxlen=self.window_size)
                self._telemetry_windows[iid].append(payload)
            
            elif topic == self.events_topic:
                self._latest_events.append(payload)
                # Keep bounded
                if len(self._latest_events) > 1000:
                    self._latest_events = self._latest_events[-500:]
                
                # Fire callbacks
                for cb in self._event_callbacks:
                    try:
                        cb(payload)
                    except Exception as e:
                        logger.error(f"Event callback error: {e}")

    def get_latest_state(self, interceptor_id: int = 0) -> Optional[dict]:
        """
        Get the most recent telemetry snapshot for an interceptor.
        
        Returns:
            Latest state dict or None if no data available
        """
        with self._lock:
            window = self._telemetry_windows.get(interceptor_id)
            if window and len(window) > 0:
                return window[-1]
        return None

    def get_state_window(self, interceptor_id: int = 0, n: int = 10) -> list:
        """
        Get the last N telemetry snapshots for temporal processing (LSTM input).
        
        Args:
            interceptor_id: Interceptor to query
            n: Number of recent snapshots to return
            
        Returns:
            List of state dicts (oldest first)
        """
        with self._lock:
            window = self._telemetry_windows.get(interceptor_id)
            if window:
                return list(window)[-n:]
        return []

    def get_latest_events(self, n: int = 10) -> list:
        """Get the N most recent engagement events."""
        with self._lock:
            return self._latest_events[-n:]

    def register_event_callback(self, callback: Callable):
        """Register a callback to be invoked for each engagement event."""
        self._event_callbacks.append(callback)

    def ingest_from_buffer(self, buffer: list):
        """
        Ingest messages from an in-memory buffer (for non-Kafka mode).
        
        This allows the consumer to work with the producer's in-memory
        fallback buffer when Kafka is not available.
        
        Args:
            buffer: List of message dicts from TelemetryStreamer.get_buffer()
        """
        for msg in buffer:
            if "event_type" in msg:
                self._process_message(self.events_topic, msg)
            else:
                self._process_message(self.telemetry_topic, msg)


# ─── Standalone test ───
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    consumer = TelemetryConsumer(enable_kafka=False)
    
    # Simulate ingesting from buffer
    mock_buffer = [
        {"interceptor_id": 0, "time_step": i, "distance": 10000 - i * 100}
        for i in range(50)
    ]
    mock_buffer.append({
        "interceptor_id": 0,
        "event_type": "INTERCEPT_HIT",
        "details": {"miss_distance": 2.1}
    })
    
    consumer.ingest_from_buffer(mock_buffer)
    
    latest = consumer.get_latest_state(0)
    window = consumer.get_state_window(0, n=5)
    events = consumer.get_latest_events(5)
    
    print(f"Latest state: time_step={latest['time_step']}, distance={latest['distance']}")
    print(f"Window (last 5): {[s['time_step'] for s in window]}")
    print(f"Events: {[e['event_type'] for e in events]}")
