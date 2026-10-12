"""
MQTT Ingestion Service for Electricity Consumption Forecasting

Connects to an MQTT broker, subscribes to meter reading topics, and
appends each incoming reading into a local SQLite table (swap for
Postgres/TimescaleDB at production scale — same logic applies).

Expected message format (JSON), published by each smart meter to
topic "meters/<client_id>/reading":

    {"timestamp": "2026-09-30T14:00:00", "load_kw": 3.42}

Run: python src/mqtt_ingest.py
Stops with Ctrl+C. Runs forever otherwise — in production this would
run as a systemd service or inside a Docker container.
"""

import json
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt

# ---------- Config ----------
BROKER_HOST = "test.mosquitto.org"   # swap for your real broker (e.g. HiveMQ, AWS IoT Core)
BROKER_PORT = 1883
TOPIC = "meters/+/reading"           # "+" is an MQTT wildcard: matches any client_id
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "readings.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("mqtt_ingest")


def init_db():
    """Create the raw_readings table if it doesn't exist yet."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_readings (
            client_id   TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            load_kw     REAL NOT NULL,
            received_at TEXT NOT NULL,
            PRIMARY KEY (client_id, timestamp)
        )
    """)
    conn.commit()
    conn.close()


def save_reading(client_id: str, timestamp: str, load_kw: float):
    """Insert one reading; ignore if this (client, timestamp) already exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR IGNORE INTO raw_readings (client_id, timestamp, load_kw, received_at) "
        "VALUES (?, ?, ?, ?)",
        (client_id, timestamp, load_kw, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        log.info(f"Connected to broker {BROKER_HOST}")
        client.subscribe(TOPIC)
        log.info(f"Subscribed to topic: {TOPIC}")
    else:
        log.error(f"Connection failed, reason code: {reason_code}")


def on_message(client, userdata, msg):
    """Called every time a meter publishes a new reading."""
    try:
        # topic looks like: meters/42/reading  -> extract "42" as client_id
        client_id = msg.topic.split("/")[1]
        payload = json.loads(msg.payload.decode())

        timestamp = payload["timestamp"]
        load_kw = float(payload["load_kw"])

        save_reading(client_id, timestamp, load_kw)
        log.info(f"Saved reading: client={client_id} time={timestamp} load={load_kw}kW")

    except (KeyError, ValueError, json.JSONDecodeError) as e:
        log.warning(f"Skipped malformed message on {msg.topic}: {e}")


def on_disconnect(client, userdata, reason_code, properties=None):
    log.warning(f"Disconnected (reason: {reason_code}). Auto-reconnect will retry.")


def run():
    init_db()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    # If your broker needs auth (most production brokers do):
    # client.username_pw_set("your_username", "your_password")
    # client.tls_set()  # for MQTTS (encrypted, port 8883)

    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    log.info("Listening for meter readings... (Ctrl+C to stop)")
    client.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    run()
