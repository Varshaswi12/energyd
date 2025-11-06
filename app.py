# app.py — Real-Time Energy Forecast for Hyderabad (OpenWeather + LightGBM)
from flask import Flask, render_template, jsonify
import pandas as pd
import joblib
import time
import threading
import requests
from datetime import datetime
import pytz
import os
import traceback

# ---- Flask setup ----
app = Flask(__name__)

# ---- Model & Config ----
MODEL_PATH = "energy_forecast_model.pkl"

# Load the trained model safely
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully:", MODEL_PATH)
    except Exception as e:
        print("⚠️ Error loading model:", e)
        traceback.print_exc()
        model = None
else:
    print("⚠️ Model file not found:", MODEL_PATH)
    model = None

# ---- OpenWeather API Config for Hyderabad ----
CITY_NAME = "Berlin"
CITY_ID = 2950159  # Hyderabad city ID
API_KEY = "616abd91a0bff781545c2d5cd31fa774"  # your key (ensure active)

def build_api_url(city_id, key):
    return f"https://api.openweathermap.org/data/2.5/weather?id={city_id}&appid={key}&units=metric"

API_URL = build_api_url(CITY_ID, API_KEY)

# ---- Global list to store live predictions ----
latest_predictions = []

# ---- Function to fetch live weather ----
def get_live_weather():
    """Fetch live Hyderabad weather data (includes icon + description)."""
    try:
        res = requests.get(API_URL, timeout=10)
        if res.status_code != 200:
            print("⚠️ Weather API error:", res.status_code, res.text)
            return None

        data = res.json()
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"] if "wind" in data else 0.0
        clouds = data.get("clouds", {}).get("all", 0)
        weather_main = data.get("weather", [{}])[0].get("main", "")
        weather_desc = data.get("weather", [{}])[0].get("description", "")
        weather_icon = data.get("weather", [{}])[0].get("icon", "")  # e.g. "04d"
        return {
            "temp": temp,
            "humidity": humidity,
            "wind": wind,
            "clouds": clouds,
            "main": weather_main,
            "desc": weather_desc,
            "icon": weather_icon
        }
    except Exception as e:
        print("⚠️ Error fetching weather data:", e)
        return None


# ---- Prediction logic (robust to feature mismatch) ----
def predict_energy(weather):
    """Predict energy demand using model and weather features.
       If model expects extra columns, fill them with sensible defaults."""
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    hour = now.hour
    dayofweek = now.weekday()
    month = now.month
    is_weekend = 1 if dayofweek >= 5 else 0

    # Base features we can compute live
    base_row = {
        "hour": hour,
        "dayofweek": dayofweek,
        "month": month,
        "is_weekend": is_weekend,
        "temp": weather.get("temp"),
        "humidity": weather.get("humidity"),
        "wind": weather.get("wind"),
        "clouds": weather.get("clouds"),
    }

    X_live = pd.DataFrame([base_row])

    # If model loaded and has expected feature names, build full row
    if model is not None and hasattr(model, "feature_name_"):
        expected = list(model.feature_name_)
        # create full dataframe with expected cols (order matters)
        X_full = pd.DataFrame(columns=expected, index=[0])

        # fill with values from X_live where names overlap
        for c in expected:
            if c in X_live.columns:
                X_full.at[0, c] = X_live.at[0, c]
            else:
                # sensible defaults:
                if c.lower().startswith("is_") or c.lower().endswith("_flag"):
                    X_full.at[0, c] = 0
                elif any(k in c.lower() for k in ["hour", "day", "month", "weekday"]):
                    X_full.at[0, c] = 0
                else:
                    X_full.at[0, c] = 0.0

        # convert types to numeric where possible
        for col in X_full.columns:
            X_full[col] = pd.to_numeric(X_full[col], errors="coerce").fillna(0.0)

        try:
            pred = model.predict(X_full)[0]
            return round(float(pred), 2)
        except Exception as e:
            print("⚠️ Model prediction error (shape/other):", e)
            # fallback if model still fails

    # fallback heuristic (keeps app running)
    temp = base_row.get("temp", 25.0) or 25.0
    humidity = base_row.get("humidity", 50.0) or 50.0
    wind = base_row.get("wind", 2.0) or 2.0
    pred = 20000 + (temp * 250) + (humidity * 30) - (wind * 40) + hour * 10
    return round(float(pred), 2)


# ---- Background updater ----
def update_live_data(interval_seconds=10):
    """Continuously fetch weather and generate predictions."""
    global latest_predictions
    print(f"🌦️ Starting real-time data updates every {interval_seconds} seconds...")

    while True:
        try:
            weather = get_live_weather()
            if weather:
                pred = predict_energy(weather)
                now = datetime.now(pytz.timezone("Asia/Kolkata"))

                record = {
                    "time": now.strftime("%Y-%m-%d %H:%M"),
                    "predicted": pred,
                    "temp": weather["temp"],
                    "humidity": weather["humidity"],
                    "wind": weather["wind"],
                    "clouds": weather["clouds"],
                    "weather_main": weather["main"],
                    "weather_desc": weather["desc"],
                    "weather_icon": weather["icon"]
                }

                latest_predictions.append(record)
                if len(latest_predictions) > 360:  # keep last ~1 hour
                    latest_predictions = latest_predictions[-360:]

                print(f"[{record['time']}] 🔹 Predicted: {pred} MW | 🌡️ {weather['temp']}°C | {weather['desc']}")
            else:
                print("⚠️ Skipped update: Weather data unavailable.")
        except Exception as e:
            print("❌ Error in update loop:", e)
            traceback.print_exc()

        time.sleep(interval_seconds)


# ---- Start background thread ----
threading.Thread(target=update_live_data, daemon=True).start()

# ---- Flask routes ----
@app.route("/")
def home():
    return render_template("live_dashboard.html", city=CITY_NAME)

@app.route("/data")
def data():
    return jsonify(latest_predictions)


# ---- Run Flask ----
if __name__ == "__main__":
    # debug=True but disable auto-reloader (use_reloader=False)
    app.run(debug=True, use_reloader=False)
