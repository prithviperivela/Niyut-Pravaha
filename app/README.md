# Flow-Aware Music — Streamlit Dashboard

An interactive, locally-hosted music player UI that runs the full Phase 1 → Phase 4 pipeline on demand.

## What it does

Pick a seed song → the app generates a 5–20 song queue using beam search, scores every transition with the stacked classifier (P1+P2+P3), shows the anchor-drift curve, and lets you play the actual audio of every song in the queue.

## One-time setup

From the project root:

```bash
pip install streamlit
```

You should already have the rest (`tensorflow`, `xgboost`, `scikit-learn`, `librosa`, `joblib`, `pandas`, `numpy`, `matplotlib`).

Make sure these artifacts exist before launching:

- `models/phase1_xgboost_model.pkl`
- `models/phase2_siamese_model.keras`
- `models/phase3_boundary_xgb_calibrated.joblib`
- `models/phase3_stacking_logistic.joblib`
- `models/phase3_boundary_metadata.json`
- `data/processed/clean_dataset.csv`
- `data/processed/phase1_feature_weights.csv`
- `data/processed/phase2_siamese_training_pairs.csv`
- `data/processed/phase2_feature_matrices/*.npy`
- `data/processed/phase2_fixed_matrices/*.npy`

(All of these are produced by running `phase1_csv_processing.ipynb`, `Untitled1.ipynb`, and `Untitled2.ipynb` end-to-end.)

## Run it

From the **project root** (the directory above `app/`):

```bash
streamlit run app/streamlit_app.py
```

The app opens in your browser at `http://localhost:8501`.

## First-launch latency

The first time you generate a queue, the app embeds the entire candidate pool with the Siamese encoder and caches every raw matrix it touches. Expect **30–90 seconds** for the first generation. Subsequent generations are fast (~2–5 seconds) because everything is in memory.

## Controls

| Sidebar control | What it does |
|-----------------|--------------|
| Queue length    | Number of songs in the generated queue (5–20) |
| Beam width (k)  | How many partial queues beam search keeps; higher = better, slower |
| Top-N           | Stage-1 fast-filter pool size per beam step |
| α (transition)  | Weight on pairwise transition smoothness |
| β (anchor)      | Weight on similarity to the seed song |
| γ (1−drift)     | Weight on staying inside the seed's tolerance band |

Defaults match the values used in the report: α=0.40, β=0.35, γ=0.25.

## Audio playback

If the matched MP3 exists at the `audio_path` recorded in the candidate pool, an expandable "Play audio" panel appears under each queue item. Audio playback uses the browser's native `<audio>` element via Streamlit's `st.audio()`.

## Stop the app

`Ctrl+C` in the terminal that started Streamlit.
