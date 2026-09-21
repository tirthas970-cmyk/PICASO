import io
import re
import os
import random
import tarfile
import requests
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import list_repo_files
from dotenv import load_dotenv

load_dotenv()

REPO_ID = "X-FlareNet/solar-flare-magnetogram-shards"
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("Configuration Error: 'HF_TOKEN' environment variable not set in your .env file.")

print("Fetching shard list from Hugging Face...")
repo_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)

RAW_SHARDS = sorted([f for f in repo_files if f.endswith(".tar") and "v2" in f])
IMMUTABLE_SHARDS = tuple(RAW_SHARDS)
TOTAL_SHARDS = len(IMMUTABLE_SHARDS)

print(f"Found {TOTAL_SHARDS} shards in static snapshot.")

metadata_records = []
shards_processed = 0

records_lock = threading.Lock()
progress_lock = threading.Lock()

YEAR_REGEX = re.compile(r"20\d{2}")

FILENAME_REGEX = re.compile(r"AR(\d+)_(\d{8}_\d{6})\.(npy|txt)")

def process_shard_headers(shard_name):
    global shards_processed
    stream_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{shard_name}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    year_match = YEAR_REGEX.search(shard_name)
    shard_year = int(year_match.group(0)) if year_match else None
    local_buffer = {}

    error_msg = None

    try:
        response = requests.get(stream_url, headers=headers, stream=True, timeout=(10, 60))
        if response.status_code == 200:
            with tarfile.open(fileobj=response.raw, mode="r|") as tar:
                for member in tar:
                    if member.isfile():
                        # Use search instead of match in case WebDataset prefixes paths
                        match = FILENAME_REGEX.search(member.name)

                        if match:
                            noaa_id = int(match.group(1))
                            ts_str = match.group(2)
                            ext = match.group(3)

                            base_key = f"AR{noaa_id}_{ts_str}"

                            if base_key not in local_buffer:
                                local_buffer[base_key] = {
                                    "shard_file": shard_name,
                                    "key": base_key,
                                    "noaa_id": noaa_id,
                                    "year": int(ts_str[:4]) if ts_str else shard_year,
                                    "timestamp": pd.to_datetime(ts_str, format="%Y%m%d_%H%M%S"),
                                    "label": 0
                                }

                            if ext == "txt":
                                f_obj = tar.extractfile(member)
                                if f_obj:
                                    try:
                                        local_buffer[base_key]["label"] = int(f_obj.read().decode('utf-8').strip())
                                    except Exception:
                                        pass
        else:
            error_msg = f"HTTP Status {response.status_code}"
    except Exception as e:
        error_msg = str(e)

    finally:
        if error_msg:
            print(f"\n[WARNING] Shard {shard_name} failed: {error_msg}")
        elif not local_buffer:
            print(f"\n[WARNING] Shard {shard_name} processed but 0 matching files found. Check naming structure.")

        # Flush local entries safely to main record stream
        if local_buffer:
            with records_lock:
                metadata_records.extend(local_buffer.values())

        with progress_lock:
            shards_processed += 1
            if shards_processed % 5 == 0 or shards_processed == TOTAL_SHARDS:
                print(f"Progress: [{shards_processed}/{TOTAL_SHARDS}] shards scanned. Total Records: {len(metadata_records)}")

print(f"\nLaunching 10 parallel threads to process exactly {TOTAL_SHARDS} shards...")
with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(process_shard_headers, IMMUTABLE_SHARDS)

df_meta = pd.DataFrame(metadata_records)

if not df_meta.empty:
    df_meta = df_meta[["shard_file", "key", "noaa_id", "year", "timestamp", "label"]]
    df_meta.to_csv("solar_master_metadata.csv", index=False)
    print(f"\nProcessing Complete! Metatable created with {len(df_meta)} rows.")
else:
    print("\nProcessing finished, but no records were found.")
