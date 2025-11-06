# clean_and_resample.py  (final, robust)
import os
import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64tz_dtype

BASE = os.getcwd()
RAW_UCI = os.path.join(BASE, "household_power_consumption.txt")
RAW_OPSD = os.path.join(BASE, "time_series_60min_singleindex.csv")
OUT_DIR = os.path.join(BASE, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

print("BASE:", BASE)
print("OUT_DIR:", OUT_DIR)
print()

# -------------------------
# 1) UCI household cleaning (safe parse)
# -------------------------
print("1) Loading UCI household data (will parse Date+Time)...")
try:
    df_uci = pd.read_csv(RAW_UCI, sep=';', na_values=['?'], low_memory=False)
    print("Raw UCI shape:", df_uci.shape)
except Exception as e:
    print("Error reading UCI file:", e)
    raise

# combine Date + Time; dayfirst=True because format is dd/mm/YYYY
print("Combining Date and Time into datetime index...")
df_uci['dt'] = pd.to_datetime(
    df_uci['Date'].astype(str).str.strip() + ' ' + df_uci['Time'].astype(str).str.strip(),
    dayfirst=True, errors='coerce'
)
df_uci = df_uci.drop(columns=['Date', 'Time'])
df_uci = df_uci.set_index('dt').sort_index()
print("UCI indexed shape (after parse):", df_uci.shape)

# convert numeric columns
for col in df_uci.columns:
    df_uci[col] = pd.to_numeric(df_uci[col], errors='coerce')

# resample to hourly mean (use lowercase 'h' to avoid future warning)
print("Resampling to hourly mean...")
uci_hr = df_uci.resample('h').mean()

# fill small gaps
uci_hr = uci_hr.ffill(limit=6).interpolate(limit_direction='both', limit=24)

cols_keep = [c for c in [
    'Global_active_power', 'Global_reactive_power', 'Voltage',
    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
] if c in uci_hr.columns]
uci_out = uci_hr[cols_keep].copy()
uci_out.index.name = 'utc_timestamp'
uci_clean_fn = os.path.join(OUT_DIR, "uci_hourly_cleaned.csv")
uci_out.to_csv(uci_clean_fn, float_format='%.4f')
print("Saved UCI cleaned:", uci_clean_fn)
print("UCI time span:", uci_out.index.min(), "->", uci_out.index.max())
print()

# -------------------------
# 2) OPSD cleaning (DE subset) - robust read
# -------------------------
print("2) Loading OPSD header to detect DE columns...")
with open(RAW_OPSD, 'r', encoding='utf-8') as f:
    header_line = f.readline().strip()

# split on comma; OPSD is comma-separated
header = header_line.split(',')

# find timestamp column (usually 'utc_timestamp')
ts_col_candidates = [h for h in header if 'utc_timestamp' in h]
if not ts_col_candidates:
    raise RuntimeError("Could not find timestamp column in OPSD header.")
ts_col = ts_col_candidates[0]

# find DE_ columns
de_cols = [c for c in header if c.startswith('DE_')]
print("Found DE columns:", len(de_cols))
if len(de_cols) == 0:
    print("No DE_ columns found — reading a small sample for inspection.")
    df_opsd_all = pd.read_csv(RAW_OPSD, nrows=20)
    print("Sample columns:", df_opsd_all.columns.tolist()[:60])
    raise SystemExit("Adjust script if header layout differs.")

# read only timestamp + DE columns (safe and memory-light)
usecols = [ts_col] + de_cols
print("Reading OPSD with usecols length:", len(usecols))
df_opsd = pd.read_csv(RAW_OPSD, usecols=usecols, parse_dates=[ts_col])
df_opsd = df_opsd.set_index(ts_col).sort_index()
print("OPSD (DE) loaded shape:", df_opsd.shape)

# select useful DE columns (prefer load_actual, price_day_ahead, solar_generation, wind_generation)
selected = {}
for c in df_opsd.columns:
    cl = c.lower()
    if 'load_actual' in cl and 'load' not in selected:
        selected['load'] = c
    if 'price_day_ahead' in cl and 'price' not in selected:
        selected['price'] = c
    if 'solar_generation' in cl and 'solar' not in selected:
        selected['solar'] = c
    if 'wind_generation' in cl and 'wind' not in selected:
        selected['wind'] = c

print("Selected mapping (may contain fewer keys):", selected)
chosen_cols = list(selected.values())
if not chosen_cols:
    # fallback: take first 4 DE columns if detection failed
    chosen_cols = de_cols[:4]
    print("Fallback chosen_cols:", chosen_cols)

df_de = df_opsd[chosen_cols].copy()

# rename to short names when detected
rename_map = {v: k for k, v in selected.items()}
if rename_map:
    df_de = df_de.rename(columns=rename_map)

# If index is tz-aware, convert to UTC then drop tz to make tz-naive (so it matches UCI)
if is_datetime64tz_dtype(df_de.index):
    try:
        df_de.index = df_de.index.tz_convert('UTC').tz_localize(None)
        print("Converted OPSD index from tz-aware to tz-naive (UTC).")
    except Exception as e:
        # fallback: convert index to naive datetimes via Python datetimes
        print("Warning converting tz-aware index; using fallback to strip tz:", e)
        df_de.index = pd.DatetimeIndex([ts.tz_convert('UTC').replace(tzinfo=None) if getattr(ts, 'tzinfo', None) else ts for ts in df_de.index])

# fill missing values reasonably
df_de = df_de.interpolate(limit=6).ffill().bfill()

# ensure hourly index and save (use lowercase 'h')
df_de = df_de.resample('h').mean()
opsd_out_fn = os.path.join(OUT_DIR, "opsd_de_hourly.csv")
df_de.to_csv(opsd_out_fn, float_format='%.4f')
print("Saved OPSD DE cleaned:", opsd_out_fn)
print("OPSD time span:", df_de.index.min(), "->", df_de.index.max())
print()

# -------------------------
# 3) Attempt to trim to common range (if overlap); skip safely otherwise
# -------------------------
print("3) Attempting to determine common date range (if any)...")
uci_min, uci_max = (uci_out.index.min(), uci_out.index.max())
opsd_min, opsd_max = (df_de.index.min(), df_de.index.max())
print("UCI range:", uci_min, "->", uci_max)
print("OPSD range:", opsd_min, "->", opsd_max)

# if either side has NaT, skip
if pd.isna(uci_min) or pd.isna(opsd_min):
    print("Could not determine common range (NaT present) — skipping trim.")
else:
    start = max(uci_min, opsd_min)
    end = min(uci_max, opsd_max)
    if start < end:
        print("Common range:", start, "->", end)
        uci_trim = uci_out.loc[start:end]
        opsd_trim = df_de.loc[start:end]
        uci_trim.to_csv(os.path.join(OUT_DIR, "uci_hourly_cleaned_trimmed.csv"), float_format='%.4f')
        opsd_trim.to_csv(os.path.join(OUT_DIR, "opsd_de_hourly_trimmed.csv"), float_format='%.4f')
        print("Saved trimmed files to data/processed/")
    else:
        print("No overlapping date range found — skipping trimmed files.")

print("\nALL DONE.")
