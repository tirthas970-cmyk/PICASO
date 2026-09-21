import os
import drms
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

JSOC_EMAIL = os.getenv("JSOC_EMAIL")
METADATA_PATH = "solar_master_metadata.csv"
OUTPUT_PATH = "magnetograms_with_sharp_features.csv"

if not JSOC_EMAIL:
    raise ValueError("Configuration Error: 'JSOC_EMAIL' environment variable not set in your .env file.")


try:
    df_meta = pd.read_csv(METADATA_PATH)
    df_meta['timestamp'] = pd.to_datetime(df_meta['timestamp'])
    print(f"✅ Loaded existing master metatable with {len(df_meta)} rows.")
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find '{METADATA_PATH}'. Please run the previous data scripts first!")

target_years = sorted(df_meta["year"].dropna().unique().astype(int))
unique_noaa_ids = set(df_meta["noaa_id"].dropna().unique().astype(int))
print(f"Detected {len(unique_noaa_ids)} unique NOAA Active Regions across years: {target_years}")

client = drms.Client(email=JSOC_EMAIL)

print("\nMapping NOAA IDs to HARPNUMs using sparse time sampling...")
noaa_to_harp = {}

for yr in target_years:
    map_query = f"hmi.Mharp_720s[][{yr}.01.01_00:00:00_TAI-{yr}.12.31_23:59:59_TAI@5d]"
    try:
        df_map_yr = client.query(map_query, key="HARPNUM, NOAA_ARS")
        if df_map_yr is not None and not df_map_yr.empty:
            df_map_yr = df_map_yr.dropna(subset=["NOAA_ARS"])
            for _, row in df_map_yr.iterrows():
                harp = int(row["HARPNUM"])
                noaa_str = str(row["NOAA_ARS"]).replace("[", "").replace("]", "").replace("'", "")
                for n_id in noaa_str.split(","):
                    if n_id.strip().isdigit():
                        noaa_val = int(n_id.strip())
                        if noaa_val in unique_noaa_ids:
                            noaa_to_harp[noaa_val] = harp
            print(f"    [Success] Mapped Year {yr} (Found {len(noaa_to_harp)} matches so far)")
    except Exception as e:
        continue

df_meta["harp_id"] = df_meta["noaa_id"].map(noaa_to_harp)
target_harps = sorted(df_meta["harp_id"].dropna().unique().astype(int))
print(f"Successfully matched {len(target_harps)} unique indexed HARPNUM clusters.")


KEYWORDS = [
    "T_REC", "HARPNUM", "NOAA_AR",
    "USFLUX", "MEANALP", "TOTUSJH", "TOTUSJZ", "SAVNCPP",
    "TOTPOT", "MEANSHR", "MEANGAM", "MEANJZH", "MEANJZD",
    "MEANDHA", "R_VALUE", "AREA_ACR", "SHRGT45", "MEANPOT", "ABS_NJZH"
]
sharp_list = []

print("\nStarting 4-hour cadence SHARP data download for all parameters...")
for yr in target_years:
    print(f" -> Processing Year {yr}:")

    year_harps = df_meta[df_meta["year"] == yr]["harp_id"].dropna().unique().astype(int)

    for month in range(1, 13):
        start_time = f"{yr}.{month:02d}.01_00:00:00_TAI"
        days = 28 if month == 2 else 31

        harp_batches = [year_harps[i:i + 40] for i in range(0, len(year_harps), 40)]

        for batch in harp_batches:
            harp_str = ",".join(map(str, batch))
            if not harp_str:
                continue

            query_str = f"hmi.sharp_720s[{harp_str}][{start_time}/{days}d@4h]"

            try:
                df_batch = client.query(query_str, key=",".join(KEYWORDS))
                if df_batch is not None and not df_batch.empty:
                    sharp_list.append(df_batch)
                    print(f"    [Success] Got {len(df_batch)} rows for {yr}-{month:02d} | HARPs [{harp_str[:15]}...]")
            except Exception as e:
                print(f"    [Warning] Failed sub-chunk {yr}-{month:02d} for HARPs [{harp_str[:15]}...]: {e}")
                continue

if not sharp_list:
    raise RuntimeError("Zero data records successfully returned via optimized query pipeline.")


print("\nCombining data and building final matrix...")
df_sharp_all = pd.concat(sharp_list, ignore_index=True)

df_sharp_all = df_sharp_all.dropna(subset=["HARPNUM"])
df_sharp_all["HARPNUM"] = df_sharp_all["HARPNUM"].astype(int)

df_sharp_all["jsoc_time"] = pd.to_datetime(
    df_sharp_all["T_REC"].str.replace("_TAI", "", regex=False),
    format="%Y.%m.%d_%H:%M:%S"
)
df_sharp_all["jsoc_year"] = df_sharp_all["jsoc_time"].dt.year

df_sharp_all["join_time"] = df_sharp_all["jsoc_time"].dt.round("1h")
df_meta["join_time"] = df_meta["timestamp"].dt.round("1h")

print("Merging features to base metatable...")
df_final = pd.merge(
    df_meta,
    df_sharp_all,
    left_on=["harp_id", "year", "join_time"],
    right_on=["HARPNUM", "jsoc_year", "join_time"],
    how="left"
)

df_final = df_final.drop(columns=["join_time", "T_REC", "HARPNUM", "jsoc_time", "jsoc_year", "harp_id"], errors="ignore")

df_final.to_csv(OUTPUT_PATH, index=False)
print(f"\n🎉 Pipeline complete! Multimodal matrix successfully exported. Final Shape: {df_final.shape}")
