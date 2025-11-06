# visualize_results.py
import pandas as pd
import matplotlib.pyplot as plt
import os

BASE = os.getcwd()
file = os.path.join(BASE, "data", "processed", "robust_dispatch_results.csv")

# Load results
df = pd.read_csv(file)
print("Loaded:", file)
print(df.head())

# --- Plot 1: Energy mix over time ---
plt.figure(figsize=(12, 6))
plt.stackplot(
    df['hour'],
    df['solar_used'],
    df['wind_used'],
    df['gas_used'],
    labels=['Solar', 'Wind', 'Gas'],
    alpha=0.8
)
plt.title("Robust Energy Dispatch Plan (7 Days)")
plt.xlabel("Hour")
plt.ylabel("Generation (MWh)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

# --- Plot 2: Battery State of Charge ---
plt.figure(figsize=(10, 4))
plt.plot(df['hour'], df['soc'], color='purple', linewidth=2)
plt.title("Battery State of Charge (SOC)")
plt.xlabel("Hour")
plt.ylabel("Energy Stored (MWh)")
plt.grid(True)
plt.tight_layout()
plt.show()
