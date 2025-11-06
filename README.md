# ⚡ European Energy Demand Prediction (Time-Series 60-Min Dataset)

## 🧩 Project Overview
This project predicts **hourly (60-minute) household energy demand** using a European **time-series power consumption dataset**.  
The aim is to forecast future electricity demand based on historical consumption patterns, helping improve **energy efficiency** and **grid stability**.

We used a **deep learning (LSTM)** model trained on the **Time Series 60-Min Household Power Consumption Dataset** to accurately predict future energy usage.

---

## 📊 Dataset Information

**Dataset Used:**  
👉 [Time Series 60-Minute and Household Power Consumption Dataset (Kaggle)](https://www.kaggle.com/datasets/taranvee/time-series-60-min-and-household-power-consumption)  

**About the Dataset:**
- **Source:** Kaggle  
- **Region:** Europe (Household data recorded in France 🇫🇷)  
- **Time Interval:** 60 minutes (hourly)  
- **Duration:** 2006–2010  
- **Attributes:**
  - `DateTime` — Timestamp  
  - `Global_active_power (kW)` — Household active power consumption  
  - `Global_reactive_power (kW)` — Reactive power  
  - `Voltage (V)` — Average voltage  
  - `Global_intensity (A)` — Average current  
  - `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3` — Energy sub-metering values  

---

## 🧰 Tools and Technologies Used
- **Language:** Python 🐍  
- **Framework:** Flask (for web deployment)  
- **Libraries:**
  - `pandas`, `numpy` → Data cleaning and resampling  
  - `matplotlib`, `seaborn` → Visualization  
  - `scikit-learn` → Feature scaling and metrics  
  - `tensorflow`, `keras` → LSTM model for time series prediction  
- **API:** OpenWeatherMap API (for adding weather-based prediction context)  
- **Deployment:** Netlify (frontend) and Render/Localhost (Flask backend)

---

## 🤖 Machine Learning Approach

1. **Data Preprocessing**
   - Loaded and parsed timestamps.
   - Ensured data is at 60-minute intervals.
   - Normalized features using `MinMaxScaler`.
   - Split data into training and testing sets.

2. **Model Used:**  
   **LSTM (Long Short-Term Memory)**  
   - Captures sequential patterns in energy demand.
   - Input: Last 24 hourly readings.
   - Output: Next-hour energy consumption.

3. **Evaluation Metrics**
   - **MAE (Mean Absolute Error)**  
   - **RMSE (Root Mean Square Error)**  
   - **R² Score**

---

## ⚙️ Project Workflow

1. Load the Time Series 60-Min dataset  
2. Preprocess and clean missing or irregular entries  
3. Train the LSTM model on sequential data  
4. Build Flask API for model inference  
5. Integrate web interface for displaying predictions  

---

## 📈 Sample Output

| Date       | Hour | Predicted Energy (kW) |
|-------------|------|----------------------|
| 2025-11-01  | 10:00 | 1.48 |
| 2025-11-01  | 11:00 | 1.51 |
| 2025-11-01  | 12:00 | 1.54 |

---

## 🌍 Results and Insights

✅ Captures daily and weekly power usage patterns  
✅ Accurate hourly prediction with LSTM  
✅ Can be expanded to multi-household or weather-linked prediction  

---

## 🚀 Future Scope

- Include temperature, humidity, and weather effects  
- Extend to multi-country European dataset  
- Build an interactive dashboard for live forecasts  

---



## 📎 References

- [Time Series 60-Min and Household Power Consumption Dataset – Kaggle](https://www.kaggle.com/datasets/taranvee/time-series-60-min-and-household-power-consumption)  
- [OpenWeatherMap API](https://openweathermap.org/api)  

---

⭐ *If you found this project helpful, please star it on GitHub!* ⭐
