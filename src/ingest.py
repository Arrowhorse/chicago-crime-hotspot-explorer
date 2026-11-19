import pandas as pd
import numpy as np
import requests
from pathlib import Path

# The same City of Chicago dataset you used in Cell 1, accessed via its live Socrata API
SOCRATA_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"

# Fixed to UPPERCASE to match how Chicago stores Primary Type (fixes the Cell 3/6 empty-dataframe bug)
TOP_CRIMES = ("THEFT", "BATTERY", "CRIMINAL DAMAGE", "NARCOTICS", "ASSAULT")


def fetch_recent_crimes(limit: int = 100_000) -> pd.DataFrame:
    """Fetches the most recent Chicago crime records for the top 5 categories."""
    crime_list = ",".join(f"'{c}'" for c in TOP_CRIMES)
    params = {
        "$select": "id,date,primary_type,location_description,arrest,domestic,community_area,latitude,longitude",
        "$where": f"primary_type IN ({crime_list}) AND latitude IS NOT NULL AND community_area IS NOT NULL",
        "$order": "date DESC",
        "$limit": limit,
    }
    print(f"Fetching up to {limit:,} recent records from Chicago Data Portal...")
    response = requests.get(SOCRATA_URL, params=params, timeout=60)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans data types and builds temporal + cyclical features."""
    df = df.copy()

    # Convert string columns from JSON into proper datetime, numeric, and boolean types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["latitude", "longitude", "community_area"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["arrest"] = df["arrest"].astype(str).str.lower() == "true"
    df["domestic"] = df["domestic"].astype(str).str.lower() == "true"

    # Drop any rows with missing coordinates, dates, or community areas (like Cell 2)
    df = df.dropna(subset=["date", "primary_type", "latitude", "longitude", "community_area"])
    df["community_area"] = df["community_area"].astype(int)

    # Extract standard time features (like Cells 2 & 4)
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.day_of_week.isin([5, 6]).astype(int)
    df["month"] = df["date"].dt.month

    # Cyclical encoding so 23:00 and 00:00 are mathematically close for ML models
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    return df


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    raw_df = fetch_recent_crimes(limit=100_000)
    clean_df = clean_and_engineer_features(raw_df)

    output_path = Path("data/chicago_crimes_recent.parquet")
    clean_df.to_parquet(output_path, index=False)
    print(f"Success! Saved {len(clean_df):,} cleaned records to {output_path}")
    print(f"Date range: {clean_df['date'].min()} to {clean_df['date'].max()}")