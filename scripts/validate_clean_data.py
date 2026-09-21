import os
import pandas as pd

METADATA_PATH = "solar_master_metadata.csv"

def run_health_report(df, stage_name="INITIAL"):
    """Prints a structured diagnostics report on the metatable health."""
    print("\n" + "="*45)
    print(f" 🔍 MASTER METATABLE HEALTH REPORT [{stage_name}] ")
    print("="*45)

    null_counts = df.isnull().sum()
    print(f"Missing Values Breakdown:\n{null_counts.to_string()}\n")

    duplicate_keys = df.duplicated(subset=['key']).sum()
    print(f"👥 Duplicate Sample Keys Found: {duplicate_keys}")
    if duplicate_keys > 0 and stage_name == "INITIAL":
        print("   ⚠️ Warning: Some keys are duplicated. Shards contain overlapping frames.")

    unique_labels = df['label'].unique()
    print(f"🏷️ Unique Flare Labels Present: {unique_labels}")
    invalid_labels = df[~df['label'].isin([0, 1, 2])]
    if not invalid_labels.empty:
        print(f"   ❌ Critical: Found {len(invalid_labels)} rows with invalid labels outside of [0, 1, 2]!")
    else:
        print("   ✅ Flare classification labels look stable.")

    df['parsed_time'] = pd.to_datetime(df['timestamp'], errors='coerce')
    unparsed_times = df['parsed_time'].isna().sum()
    print(f"📅 Corrupted/Unparseable Timestamps: {unparsed_times}")

    mismatched_years = df[df['parsed_time'].dt.year != df['year']]
    print(f"🔄 Year Mismatches (Parsed Time vs Year Column): {len(mismatched_years)}")

    invalid_noaa = df[(df['noaa_id'] <= 0) | (df['noaa_id'].isna())]
    print(f"☀️ Invalid/Zero NOAA IDs Found: {len(invalid_noaa)}")

    print("\n" + "-"*45)
    print(" DATASET PROFILE SUMMARY ")
    print("-"*45)
    print(f"📊 Samples per Year:\n{df['year'].value_counts().sort_index().to_string()}")
    print(f"\n🔥 Samples per Flare Class:\n{df['label'].value_counts().sort_index().to_string()}")
    print("="*45 + "\n")

def main():
    try:
        df_meta = pd.read_csv(METADATA_PATH)
        print(f"✅ Loaded dataset with {len(df_meta)} records.")
    except FileNotFoundError:
        print(f"❌ Error: '{METADATA_PATH}' not found. Please run the metadata indexer first.")
        return

    run_health_report(df_meta, stage_name="PRE-CLEANUP")

    duplicate_count = df_meta.duplicated(subset=['key']).sum()
    if duplicate_count > 0:
        print(f"🧹 Commencing Sanitation: Dropping {duplicate_count} duplicates...")
        
        df_clean = df_meta.drop_duplicates(subset=['key'], keep='first')
        
        df_clean.to_csv(METADATA_PATH, index=False)
        
        print("\n🎉 CLEANUP COMPLETE")
        print(f"Dropped rows:        {len(df_meta) - len(df_clean)}")
        print(f"Final unique rows:   {len(df_clean)}")
        
        run_health_report(df_clean, stage_name="POST-CLEANUP")
    else:
        print("✅ No duplicate keys found. Dataset is already clean!")

if __name__ == "__main__":
    main()
