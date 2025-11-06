# forecast_shortterm.py
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

BASE = os.getcwd()
DATA = os.path.join(BASE, "data", "processed", "opsd_de_hourly.csv")

print("Loading:", DATA)
df = pd.read_csv(DATA, parse_dates=['utc_timestamp'], index_col='utc_timestamp')
print("Shape:", df.shape)
print(df.head())

# 1. Keep target = load (actual electricity demand)
target_col = 'load'
if target_col not in df.columns:
    raise ValueError(f"'{target_col}' column not found. Available: {df.columns.tolist()}")

# Fill missing
df = df.interpolate().ffill().bfill()

# 2. Create time-based features
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# 3. Lag features (previous hour values)
for lag in [1, 2, 3, 6, 12, 24]:
    df[f'lag_{lag}'] = df[target_col].shift(lag)

# 4. Rolling window features
for window in [3, 6, 12, 24]:
    df[f'roll_mean_{window}'] = df[target_col].shift(1).rolling(window).mean()
    df[f'roll_std_{window}'] = df[target_col].shift(1).rolling(window).std()

# Drop missing values from lagging
df = df.dropna()

# 5. Train/test split (last 10% as test)
split_idx = int(len(df) * 0.9)
train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# 6. LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# 7. Predictions
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"✅ Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"R²:  {r2:.3f}")

# 8. Plot results
plt.figure(figsize=(10, 4))
plt.plot(y_test.values[:200], label="Actual", linewidth=2)
plt.plot(preds[:200], label="Predicted", linewidth=2)
plt.title("Short-term Load Forecast (First 200 Hours of Test Set)")
plt.xlabel("Time")
plt.ylabel("Load (MW)")
plt.legend()
plt.tight_layout()
plt.show()

# 9. Save model + predictions
out_csv = os.path.join(BASE, "data", "processed", "shortterm_predictions.csv")
pd.DataFrame({"actual": y_test, "predicted": preds}, index=y_test.index).to_csv(out_csv)
print("Saved predictions to:", out_csv)

import joblib
joblib.dump(model, "energy_forecast_model.pkl")
print("✅ Model saved as energy_forecast_model.pkl")
