import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

CLUSTER_FEATURES = ["latitude", "longitude", "hour_sin", "hour_cos", "is_weekend"]


def train_spatiotemporal_clusters(df: pd.DataFrame, n_clusters: int = 8) -> tuple[pd.DataFrame, dict]:
    """Trains MiniBatchKMeans on standardized location + cyclical time features."""
    df = df.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTER_FEATURES])

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=4096,
        n_init=10,
    )
    df["cluster_id"] = model.fit_predict(X_scaled)

    # Sample 10,000 points for fast silhouette validation (avoids O(N^2) memory hang)
    sample_idx = np.random.RandomState(42).choice(len(X_scaled), size=min(10_000, len(X_scaled)), replace=False)
    sil_score = float(silhouette_score(X_scaled[sample_idx], df["cluster_id"].values[sample_idx]))

    # Generate human-readable labels for each cluster based on dominant crime & peak hour
    cluster_names = {}
    for cid, group in df.groupby("cluster_id"):
        top_crime = group["primary_type"].mode()[0].title()
        peak_hour = int(group["hour"].mode()[0])
        time_label = (
            "Late Night" if peak_hour < 5 or peak_hour >= 22
            else "Morning" if peak_hour < 12
            else "Afternoon" if peak_hour < 17
            else "Evening"
        )
        cluster_names[int(cid)] = f"Cluster {cid}: {time_label} {top_crime} (~{peak_hour:02d}:00)"

    df["cluster_label"] = df["cluster_id"].map(cluster_names)

    artifact = {
        "scaler": scaler,
        "model": model,
        "features": CLUSTER_FEATURES,
        "cluster_names": cluster_names,
        "silhouette_score": round(sil_score, 4),
    }
    return df, artifact


def compute_community_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates 30-day crime surge Z-scores per Chicago Community Area."""
    max_date = df["date"].max()
    recent_cutoff = max_date - pd.Timedelta(days=30)

    # Create 30-day period buckets counting backward from max_date
    df = df.copy()
    df["days_ago"] = (max_date - df["date"]).dt.days
    df["period_30d"] = df["days_ago"] // 30

    # Exclude incomplete oldest bucket
    max_complete_period = df["period_30d"].max() - 1
    complete_df = df[df["period_30d"] <= max_complete_period]

    period_counts = (
        complete_df.groupby(["community_area", "period_30d"])
        .size()
        .unstack(fill_value=0)
    )

    recent_counts = period_counts[0] if 0 in period_counts.columns else pd.Series(0, index=period_counts.index)
    historical = period_counts.drop(columns=[0], errors="ignore")

    hist_mean = historical.mean(axis=1)
    hist_std = historical.std(axis=1).replace(0, np.nan)

    z_score = ((recent_counts - hist_mean) / hist_std).fillna(0.0)

    # Top crime & arrest rate in the last 30 days per community area
    recent_df = df[df["date"] >= recent_cutoff]
    top_crime = recent_df.groupby("community_area")["primary_type"].agg(lambda s: s.mode()[0] if len(s) else "N/A")
    arrest_rate = recent_df.groupby("community_area")["arrest"].mean() * 100
    center_lat = df.groupby("community_area")["latitude"].mean()
    center_lon = df.groupby("community_area")["longitude"].mean()

    risk_df = pd.DataFrame({
        "community_area": period_counts.index,
        "recent_30d_incidents": recent_counts.values,
        "baseline_30d_avg": hist_mean.round(1).values,
        "surge_z_score": z_score.round(2).values,
    }).set_index("community_area")

    risk_df["top_crime_30d"] = top_crime
    risk_df["arrest_rate_pct"] = arrest_rate.round(1)
    risk_df["latitude"] = center_lat
    risk_df["longitude"] = center_lon
    risk_df["risk_status"] = pd.cut(
        risk_df["surge_z_score"],
        bins=[-np.inf, 0.5, 1.5, np.inf],
        labels=["Normal / Declining", "Elevated", "High Surge"],
    ).astype(str)

    return risk_df.reset_index().dropna(subset=["latitude", "longitude"])


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    input_path = Path("data/chicago_crimes_recent.parquet")
    df = pd.read_parquet(input_path)

    print("Training spatio-temporal hotspot clusters...")
    enriched_df, model_artifact = train_spatiotemporal_clusters(df, n_clusters=8)

    print("Computing 30-day Community Area anomaly Z-scores...")
    risk_df = compute_community_risk_scores(enriched_df)

    joblib.dump(model_artifact, "models/hotspot_model.joblib")
    enriched_df.to_parquet("data/chicago_crimes_enriched.parquet", index=False)
    risk_df.to_parquet("data/community_risk.parquet", index=False)

    print(f"Done! Silhouette Score: {model_artifact['silhouette_score']}")
    print("\nDiscovered Spatio-Temporal Regimes:")
    for name in model_artifact["cluster_names"].values():
        print(f"  - {name}")
    print("\nTop 5 Community Areas by 30-Day Surge Z-Score:")
    print(risk_df.sort_values("surge_z_score", ascending=False)[["community_area", "recent_30d_incidents", "baseline_30d_avg", "surge_z_score", "risk_status"]].head(5).to_string(index=False))