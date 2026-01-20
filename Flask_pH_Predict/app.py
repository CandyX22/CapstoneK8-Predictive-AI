import numpy as np
import pandas as pd
import joblib
import requests
import os
import json
import paho.mqtt.client as mqtt

from flask import Flask, jsonify
from keras.models import load_model
from datetime import datetime, timedelta
from supabase import create_client

# ================= CONFIG =================
MODEL_PATH = "models/lstm1m_ph_model.h5"
SCALER_PATH = "models/scaler1m_ph.pkl"

SUPABASE_URL = "https://vltxzjfjlnzdmgszdciv.supabase.co"
SUPABASE_KEY = "sb_publishable_RjomUSwf9oGMIgnG4Kl0yA_pOUF51r8"
# API-DE-For Get
REPLIT_API_URL = "https://240d6c5a-085e-4090-9e99-77c9e41ddc06-00-1p1eomiap7bc6.picard.replit.dev/api/get_ph"

# MQTT CONFIG (HiveMQ Cloud)
MQTT_BROKER = "3f8165ca59d840d9bc964c540d1b792e.s1.eu.hivemq.cloud"
MQTT_PORT   = 8883
MQTT_USER   = "Hydroponic"
MQTT_PASS   = "Hydro1234"
MQTT_TOPIC  = "iot/actuator/pompa"

WINDOW_SIZE = 120
PREDICT_MINUTES = 20

PH_MIN = 5.5
PH_MAX = 6.5

MIN_OUT_OF_RANGE_COUNT = 5

app = Flask(__name__)

# ================= INIT =================
print("Loading model & scaler...")
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
print("✅ Model & scaler loaded")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Supabase connected")

# MQTT Client (TLS + Auth)
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
mqtt_client.tls_set()  # enable SSL/TLS
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

print("📡 MQTT connected (HiveMQ Cloud)")


# ================= ROUTES =================

@app.route("/fetch-sensor", methods=["GET"])
def fetch_sensor_data():
    try:
        res = requests.get(REPLIT_API_URL, timeout=10)
        raw_json = res.json()

        if "data" not in raw_json:
            return jsonify({"error": "Invalid data format"}), 400

        sensor_data = [
            {"ph": float(i["ph"]), "timestamp": i["timestamp"]}
            for i in raw_json["data"]
            if "ph" in i and "timestamp" in i
        ]

        supabase.table("dataset_sensor").delete().neq("id", 0).execute()
        supabase.table("dataset_sensor").insert(sensor_data).execute()

        return jsonify({
            "status": "success",
            "total_inserted": len(sensor_data)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/run-prediction", methods=["GET"])
def run_prediction():
    response = supabase.table("dataset_sensor") \
        .select("timestamp, ph") \
        .order("timestamp") \
        .execute()

    if len(response.data) < WINDOW_SIZE:
        return jsonify({"error": "Not enough data"}), 400

    df = pd.DataFrame(response.data)
    ph_values = df["ph"].values.reshape(-1, 1)

    window = scaler.transform(ph_values[-WINDOW_SIZE:])
    window = window.reshape(1, WINDOW_SIZE, 1)

    predictions = []
    now = datetime.utcnow()

    for i in range(PREDICT_MINUTES):
        pred_scaled = model.predict(window, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0][0]

        predictions.append({
            "timestamp": (now + timedelta(minutes=i + 1)).isoformat(),
            "ph": round(float(pred), 3)
        })

        window = np.append(
            window[:, 1:, :],
            pred_scaled.reshape(1, 1, 1),
            axis=1
        )

    supabase.table("prediksi_ph").delete().neq("id", 0).execute()
    supabase.table("prediksi_ph").insert(predictions).execute()

    return jsonify({
        "status": "success",
        "total_predictions": len(predictions)
    })


@app.route("/run-control-pump", methods=["GET"])
def run_control_pump():
    """
    Evaluasi hasil prediksi pH
    - Cek 20 menit ke depan
    - Jika >= 5 prediksi di luar range → pompa ON
    - Endpoint dipanggil tiap 10 menit
    """

    res = supabase.table("prediksi_ph") \
        .select("timestamp, ph") \
        .order("timestamp") \
        .execute()

    if not res.data or len(res.data) < PREDICT_MINUTES:
        return jsonify({"error": "Prediction data incomplete"}), 400

    out_of_range_count = sum(
        1 for item in res.data
        if float(item["ph"]) < PH_MIN or float(item["ph"]) > PH_MAX
    )

    pump_status = "ON" if out_of_range_count >= MIN_OUT_OF_RANGE_COUNT else "OFF"

    mqtt_client.publish(
        MQTT_TOPIC,
        json.dumps({"pump": pump_status})
    )

    return jsonify({
        "status": "success",
        "pump": pump_status,
        "out_of_range_count": out_of_range_count,
        "threshold": MIN_OUT_OF_RANGE_COUNT,
        "evaluated_at": datetime.utcnow().isoformat()
    })


@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    res = supabase.table("prediksi_ph") \
        .select("*") \
        .order("timestamp") \
        .execute()

    return jsonify(res.data)


# ================= MAIN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
