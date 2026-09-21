import os
import glob
import io
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import astropy.units as u
import webdataset as wbd
import sunpy.map
from sunpy.net import Fido, attrs as a
from sunpy.net.hek import HEKClient
from huggingface_hub import HfApi

load_dotenv()

JSOC_EMAIL = os.getenv("JSOC_EMAIL")
HF_TOKEN = os.getenv("HF_TOKEN")

# Fail early if critical configuration secrets are missing
if not JSOC_EMAIL:
    raise ValueError("Configuration Error: 'JSOC_EMAIL' environment variable not set.")
if not HF_TOKEN:
    raise ValueError("Configuration Error: 'HF_TOKEN' environment variable not set.")

start_time = "2014-08-01T00:00:00"
end_time = "2014-09-30T23:59:59"
chunk_label = "2014_ch7"

LOCAL_TEMP_DIR = Path("./raw_fits")
LOCAL_SHARD_DIR = Path("./solar_shards")

shutil.rmtree(str(LOCAL_TEMP_DIR), ignore_errors=True)
shutil.rmtree(str(LOCAL_SHARD_DIR), ignore_errors=True)
LOCAL_TEMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_SHARD_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "X-FlareNet/solar-flare-magnetogram-shards"

print(f"Starting pipeline processing for chunk: {chunk_label}")

print("Fetching the flare log from HEK")
hek_client = HEKClient()
event_search = hek_client.search(
    a.Time(start_time, pd.to_datetime(end_time) + pd.Timedelta(days=2)),
    a.hek.EventType("FL"),
    a.hek.FL.GOESCls >= "M1.0"
)
if event_search:
    df_flares = event_search[[name for name in event_search.colnames if len(event_search[name].shape) <= 1]].to_pandas()
    df_flares['ar_noaanum'] = pd.to_numeric(df_flares['ar_noaanum'], errors='coerce')
    df_flares['event_starttime'] = pd.to_datetime(df_flares['event_starttime'])
else:
    df_flares = pd.DataFrame(columns=['ar_noaanum', 'event_starttime'])

print("Searching JSOC for SHARP Magnetograms")
search_results = Fido.search(
    a.Time(start_time, end_time),
    a.jsoc.Series("hmi.sharp_cea_720s"),
    a.jsoc.Segment("magnetogram"),
    a.Sample(14400 * u.s), # Every 4 hours
    a.jsoc.Notify(JSOC_EMAIL) # Fixed case mismatch here
)
if not search_results:
    print("No images matched criteria.")
    exit()

print(f"Downloading {len(search_results['jsoc'])} raw FITS files to local disk...")
downloaded_files = Fido.fetch(search_results, path=str(LOCAL_TEMP_DIR / "{file}"))

print("Packing files into .tar shards...")
shard_pattern = os.path.join(LOCAL_SHARD_DIR, f"shard_{chunk_label}_%06d-v2.tar")

with wbd.ShardWriter(shard_pattern, maxsize=int(2e8)) as sink: # 200MB protects Colab RAM from crashing
    for file_path in downloaded_files:
        try:
            sharp_map = sunpy.map.Map(file_path)
            noaa_string = sharp_map.meta.get('noaa_ars', '')
            try:
                sharp_ar = int(noaa_string.split(',')[0].strip()) if noaa_string else int(sharp_map.meta.get('noaa_ar', 0))
            except (ValueError, TypeError):
                sharp_ar = int(sharp_map.meta.get('noaa_ar', 0))

            if sharp_ar == 0:
                print(f"Skipping file {file_path.name}: No valid NOAA AR found.")
                Path(file_path).unlink(missing_ok=True)
                continue

            t_rec_str = sharp_map.meta.get('t_rec', '').replace('_TAI', '').replace('_', ' ')
            img_time = pd.to_datetime(t_rec_str)

            flare_match = df_flares[
                (df_flares['ar_noaanum'] == sharp_ar) &
                (df_flares['event_starttime'] > img_time - pd.Timedelta(hours=1)) & #grace period
                (df_flares['event_starttime'] <= img_time + pd.Timedelta(hours=24))
            ]
            if not flare_match.empty:
              goes_classes = [str(c).upper() for c in flare_match['fl_goescls'].tolist()]

              if any(c.startswith('X') for c in goes_classes):
                  label = 2  # Class 2: X-Class Flare
              elif any(c.startswith('M') for c in goes_classes):
                label = 1  # Class 1: M-Class Flare
              else:
                label = 0 #just in case
            else:
               label = 0

            raw_pixels = np.nan_to_num(sharp_map.data).astype(np.float32)

            TARGET_SIZE = 512
            h, w = raw_pixels.shape
            pad_h = TARGET_SIZE - h
            pad_w = TARGET_SIZE - w
            if pad_h >= 0 and pad_w >= 0:
                final_pixels = np.pad(
                    raw_pixels,
                    ((pad_h // 2, pad_h - (pad_h // 2)), (pad_w // 2, pad_w - (pad_w // 2))),
                    mode='constant', constant_values=0.0
                )
            else:
                temp_pad_h = max(0, pad_h)
                temp_pad_w = max(0, pad_w)
                padded_temp = np.pad(
                    raw_pixels,
                    ((temp_pad_h // 2, temp_pad_h - (temp_pad_h // 2)), (temp_pad_w // 2, temp_pad_w - (temp_pad_w // 2))),
                    mode='constant', constant_values=0.0
                )
                th, tw = padded_temp.shape
                final_pixels = padded_temp[
                    (th - TARGET_SIZE) // 2 : (th + TARGET_SIZE) // 2,
                    (tw - TARGET_SIZE) // 2 : (tw + TARGET_SIZE) // 2
                ]

            buffer = io.BytesIO()
            np.save(buffer, final_pixels)
            npy_bytes = buffer.getvalue()
            base_key = f"AR{sharp_ar}_{img_time.strftime('%Y%m%d_%H%M%S')}"

            sink.write({
                "__key__": base_key,
                "npy": npy_bytes,
                "txt": str(label).encode('utf-8')
            })
            Path(file_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"Error packing file {file_path}: {e}")
            Path(file_path).unlink(missing_ok=True)

print("All data packed into local shards successfully!")
print("Uploading local shards up to your private Hugging Face Repository...")
api = HfApi()
api.upload_folder(
    folder_path=str(LOCAL_SHARD_DIR),
    repo_id=REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
)
print(f"Pipeline Complete! Shards for {chunk_label} are live on Hugging Face Hub.")
