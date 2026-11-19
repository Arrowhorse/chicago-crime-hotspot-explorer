# 🚨 Chicago Crime Spatio-Temporal Hotspot & AI Explorer

An end-to-end data ingestion, machine learning, and interactive geospatial web application analyzing live public safety telemetry from the City of Chicago Data Portal.

**🔗 Live App:** *(https://chicago-crime-hotspot-explorer-5zwwmqdgyq2keqgavhwjxk.streamlit.app/)*

## Key Engineering & ML Highlights
- **Automated Data Pipeline (`src/ingest.py`):** Queries the Chicago Data Portal Socrata API for the 100,000 most recent incidents across top offense categories, applying cyclical encoding ($\sin/\cos$) to preserve circular 24-hour and 12-month continuity and storing compressed `.parquet` artifacts.
- **Spatio-Temporal Clustering (`src/train.py`):** Trains `MiniBatchKMeans` on standardized spatial coordinates (`Latitude`, `Longitude`) and cyclical temporal features to uncover distinct neighborhood-time crime regimes.
- **Rolling Anomaly Detection (`src/train.py`):** Computes 30-day rolling incident Z-scores per Chicago Community Area against historical baselines to flag statistically elevated crime surges.
- **Interactive 3D Web App & AI Briefing (`app.py`):** Built with Streamlit and PyDeck (`HexagonLayer` + `ScatterplotLayer`), featuring optional Gemini LLM synthesis to translate filtered map telemetry into executive patrol briefings.
- **CI/CD Scheduled Retraining (`.github/workflows/refresh_data.yml`):** GitHub Actions workflow refreshes data and retrains model artifacts automatically every week.

## Run Locally
```bash
git clone [https://github.com/Arrowhorse/chicago-crime-hotspot-explorer.git](https://github.com/Arrowhorse/chicago-crime-hotspot-explorer.git)
cd chicago-crime-hotspot-explorer
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/ingest.py
python src/train.py
streamlit run app.py
