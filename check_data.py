# check_data.py  -- view contents of both datasets
import pandas as pd
import os

print("\n📂 Checking datasets in this folder...\n")

# File paths (in the same directory)
uci_file = "household_power_consumption.txt"
opsd_file = "time_series_60min_singleindex.csv"

# Confirm both files exist
for file in [uci_file, opsd_file]:
    if os.path.exists(file):
        size = os.path.getsize(file) / (1024 * 1024)
        print(f"✅ Found: {file} ({size:.1f} MB)")
    else:
        print(f"❌ Missing: {file}")

# Preview 10 rows from UCI dataset
print("\n────────────────────────────")
print("🏠 UCI Household Dataset (first 10 rows)")
print("────────────────────────────")
uci_df = pd.read_csv(uci_file, sep=';', nrows=10)
print(uci_df.head(10).to_string(index=False))

# Preview 10 rows from OPSD dataset
print("\n────────────────────────────")
print("⚡ OPSD Dataset (first 10 rows)")
print("────────────────────────────")
opsd_df = pd.read_csv(opsd_file, nrows=10)
print(opsd_df.head(10).to_string(index=False))

print("\n✅ Done — both datasets previewed successfully.")
