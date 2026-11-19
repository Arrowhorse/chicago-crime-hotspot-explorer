import os
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="Chicago Crime Hotspot & AI Explorer",
    page_icon="🚨",
    layout="wide",
)


@st.cache_data
def load_data():
    crimes = pd.read_parquet("data/chicago_crimes_enriched.parquet")
    risk = pd.read_parquet("data/community_risk.parquet")

    # Enrich cluster labels with their primary Community Area so every label is unique
    dom_area = crimes.groupby("cluster_id")["community_area"].agg(lambda s: int(s.mode()[0]))
    crimes["unique_cluster"] = crimes.apply(
        lambda r: f"{r['cluster_label']} [Area {dom_area[r['cluster_id']]}]", axis=1
    )
    return crimes, risk


@st.cache_resource
def load_model():
    return joblib.load("models/hotspot_model.joblib")


def generate_briefing(filtered_df: pd.DataFrame, risk_df: pd.DataFrame, api_key: str = "") -> str:
    """Generates an executive briefing using Gemini if an API key is provided, or a smart analytical fallback."""
    top_crime = filtered_df["primary_type"].mode()[0]
    peak_hour = int(filtered_df["hour"].mode()[0])
    arrest_pct = filtered_df["arrest"].mean() * 100
    top_area = int(filtered_df["community_area"].mode()[0])
    surging = risk_df[risk_df["surge_z_score"] > 1.5].sort_values("surge_z_score", ascending=False)
    surge_text = (
        ", ".join(f"Area {int(r.community_area)} (Z={r.surge_z_score})" for _, r in surging.head(3).iterrows())
        if len(surging) > 0
        else "None above Z=1.5"
    )

    summary_context = (
        f"Filtered Incidents: {len(filtered_df):,}\n"
        f"Dominant Offense: {top_crime}\n"
        f"Peak Hour: {peak_hour:02d}:00\n"
        f"Most Affected Community Area: Area {top_area}\n"
        f"Arrest Rate: {arrest_pct:.1f}%\n"
        f"30-Day High-Surge Areas (Z > 1.5): {surge_text}"
    )

    if api_key.strip():
        try:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key.strip()}"
            prompt = (
                "You are a senior civic data scientist briefing city public safety analysts. "
                "Based on the filtered Chicago crime telemetry below, write 3 concise, actionable bullet points "
                "highlighting spatial-temporal patterns, resource allocation recommendations, and statistical anomalies:\n\n"
                f"{summary_context}"
            )
            res = requests.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=20,
            )
            res.raise_for_status()
            return res.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"*AI API unavailable ({e}). Showing analytical summary:*\n\n" + _fallback_bullets(
                top_crime, peak_hour, top_area, arrest_pct, surge_text
            )

    return _fallback_bullets(top_crime, peak_hour, top_area, arrest_pct, surge_text)


def _fallback_bullets(top_crime, peak_hour, top_area, arrest_pct, surge_text) -> str:
    return (
        f"- **Primary Concentration:** **{top_crime.title()}** is the leading offense in the selected slice, peaking around **{peak_hour:02d}:00** with highest concentration in **Community Area {top_area}**.\n"
        f"- **Enforcement Signal:** Current arrest clearance rate for this segment is **{arrest_pct:.1f}%**, indicating opportunities for targeted preventative patrol during peak hours.\n"
        f"- **Statistical Surge Alert:** Rolling 30-day Z-score monitoring flags **{surge_text}** as statistically elevated relative to historical baselines."
    )


# --- Load Artifacts ---
crimes_df, risk_df = load_data()
artifact = load_model()

# --- Header ---
st.title("🚨 Chicago Crime Spatio-Temporal Hotspot & AI Explorer")
st.markdown(
    f"Live pipeline ingesting **100,000 recent records** (`{crimes_df['date'].min().strftime('%Y-%m-%d')}` to "
    f"`{crimes_df['date'].max().strftime('%Y-%m-%d')}`) from the City of Chicago Data Portal, modeled with "
    f"**MiniBatchKMeans** clustering and **30-day rolling Z-score** anomaly detection."
)

# --- Sidebar Controls ---
st.sidebar.header("Filter Telemetry")
selected_crimes = st.sidebar.multiselect(
    "Crime Categories",
    options=sorted(crimes_df["primary_type"].unique()),
    default=sorted(crimes_df["primary_type"].unique()),
)

cluster_options = ["All Regimes"] + sorted(crimes_df["unique_cluster"].unique())
selected_cluster = st.sidebar.selectbox("Spatio-Temporal Regime", options=cluster_options)

hour_range = st.sidebar.slider("Hour of Day", min_value=0, max_value=23, value=(0, 23))
day_filter = st.sidebar.radio("Day Type", options=["All Days", "Weekdays Only", "Weekends Only"])

st.sidebar.divider()
st.sidebar.header("🤖 AI Tactical Briefing")
gemini_key = st.sidebar.text_input(
    "Gemini API Key (optional)",
    type="password",
    value=os.getenv("GEMINI_API_KEY", ""),
    help="Leave blank to use built-in statistical briefing, or paste a Gemini API key for LLM synthesis.",
)

# --- Apply Filters ---
mask = (
    crimes_df["primary_type"].isin(selected_crimes)
    & (crimes_df["hour"] >= hour_range[0])
    & (crimes_df["hour"] <= hour_range[1])
)
if selected_cluster != "All Regimes":
    mask &= crimes_df["unique_cluster"] == selected_cluster
if day_filter == "Weekdays Only":
    mask &= crimes_df["is_weekend"] == 0
elif day_filter == "Weekends Only":
    mask &= crimes_df["is_weekend"] == 1

filtered = crimes_df[mask]

if filtered.empty:
    st.warning("No incidents match your current filter combination. Broaden your filters in the sidebar.")
    st.stop()

# --- KPI Row ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filtered Incidents", f"{len(filtered):,}")
c2.metric("Arrest Rate", f"{filtered['arrest'].mean() * 100:.1f}%")
c3.metric("High-Surge Areas (Z > 1.5)", int((risk_df["surge_z_score"] > 1.5).sum()))
c4.metric("Cluster Silhouette Score", artifact["silhouette_score"])

# --- AI Briefing Box ---
with st.container(border=True):
    st.subheader("📋 Tactical Analyst Briefing")
    if st.button("Generate Fresh Briefing", type="primary") or True:
        st.markdown(generate_briefing(filtered, risk_df, gemini_key))

# --- Map + Hourly Chart ---
col_map, col_chart = st.columns([1.4, 1])

with col_map:
    st.subheader("3D Incident Density & Surge Zones")
    # Sample up to 25k points for smooth browser WebGL rendering
    map_sample = filtered.sample(n=min(25_000, len(filtered)), random_state=42)

    hex_layer = pdk.Layer(
        "HexagonLayer",
        data=map_sample[["longitude", "latitude"]],
        get_position="[longitude, latitude]",
        radius=250,
        elevation_scale=8,
        elevation_range=[0, 1000],
        pickable=True,
        extruded=True,
        coverage=0.85,
    )

    surge_areas = risk_df[risk_df["surge_z_score"] > 1.0].copy()
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=surge_areas,
        get_position="[longitude, latitude]",
        get_color="[255, 50, 50, 180]",
        get_radius=900,
        pickable=True,
    )

    view_state = pdk.ViewState(latitude=41.84, longitude=-87.68, zoom=10.2, pitch=45)
    st.pydeck_chart(
        pdk.Deck(
            layers=[hex_layer, scatter_layer],
            initial_view_state=view_state,
            tooltip={"text": "Community Area: {community_area}\n30d Surge Z-Score: {surge_z_score}\nTop Crime: {top_crime_30d}"},
        )
    )

with col_chart:
    st.subheader("Hourly Crime Rhythm by Category")
    hourly = filtered.groupby(["hour", "primary_type"]).size().unstack(fill_value=0)
    st.area_chart(hourly, height=260)

    st.subheader("🚨 Top Community Area 30-Day Surges")
    top_surges = risk_df.sort_values("surge_z_score", ascending=False)[
        ["community_area", "recent_30d_incidents", "baseline_30d_avg", "surge_z_score", "top_crime_30d", "risk_status"]
    ].head(8)
    st.dataframe(top_surges, use_container_width=True, hide_index=True)