"""
Flow-Aware Music Transition System — Streamlit Dashboard
Run from project root:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
#  Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
EPS = 1e-6
K_WINDOW = 12

# --------------------------------------------------------------------------------------
#  Page config + global styles
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Flow-Aware Music", page_icon="🎧", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); color: #f5f5f5; }
    h1, h2, h3, h4 { color: #f5f5f5 !important; }
    .seed-card { background: rgba(255,255,255,0.06); border-radius: 16px; padding: 20px; margin-bottom: 16px;
                 border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(10px); }
    .queue-card { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 12px 16px; margin-bottom: 8px;
                  border-left: 3px solid #1aaf5d; transition: all .2s; }
    .queue-card:hover { background: rgba(255,255,255,0.08); transform: translateX(2px); }
    .queue-num   { color: #aaa; font-size: 12px; }
    .queue-title { color: #fff; font-size: 16px; font-weight: 600; }
    .queue-artist{ color: #bbb; font-size: 13px; }
    .score-pill  { display: inline-block; padding: 2px 10px; border-radius: 12px; background: #1aaf5d; color: #fff;
                   font-size: 12px; margin-left: 8px; }
    .score-mid   { background: #f4a261; }
    .score-low   { background: #d33; }
    .stMetric { background: rgba(255,255,255,0.04); padding: 12px; border-radius: 12px; }
    .footer-note { color: #888; font-size: 11px; text-align: center; padding: 20px; }
    </style>
    """, unsafe_allow_html=True
)

# --------------------------------------------------------------------------------------
#  Cached loaders
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Phase 1 model and weights…")
def load_p1():
    p1_model = joblib.load(MODEL_DIR / "phase1_xgboost_model.pkl")
    w_df = pd.read_csv(DATA_DIR / "phase1_feature_weights.csv")
    W_MAP = dict(zip(w_df["feature"], w_df["normalized_weight"]))
    P1_W_VEC = np.array([
        W_MAP["diff_danceability"], W_MAP["diff_energy"], W_MAP["diff_loudness"],
        W_MAP["diff_speechiness"], W_MAP["diff_acousticness"], W_MAP["diff_instrumentalness"],
        W_MAP["diff_liveness"], W_MAP["diff_valence"], W_MAP["diff_tempo"],
        W_MAP["diff_mode"], W_MAP["diff_key_fifths"],
    ])
    return p1_model, P1_W_VEC

@st.cache_resource(show_spinner="Loading Phase 2 Siamese encoder…")
def load_p2_encoder():
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    INPUT_SHAPE = (128, 30); EMB = 128
    inp = layers.Input(shape=INPUT_SHAPE)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(EMB, activation="relu")(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    encoder = Model(inp, x, name="phase2_encoder")
    src_in = layers.Input(shape=INPUT_SHAPE); tgt_in = layers.Input(shape=INPUT_SHAPE)
    dist = layers.Lambda(lambda t: tf.sqrt(tf.reduce_sum(tf.square(t[0]-t[1]), axis=1, keepdims=True)))(
        [encoder(src_in), encoder(tgt_in)]
    )
    siamese = Model([src_in, tgt_in], dist)
    siamese.load_weights(MODEL_DIR / "phase2_siamese_model.keras")
    return encoder

@st.cache_resource(show_spinner="Loading Phase 3 boundary classifier and stacker…")
def load_p3():
    calib = joblib.load(MODEL_DIR / "phase3_boundary_xgb_calibrated.joblib")
    stack = joblib.load(MODEL_DIR / "phase3_stacking_logistic.joblib")
    meta  = json.load(open(MODEL_DIR / "phase3_boundary_metadata.json"))
    return calib, stack, meta

@st.cache_data(show_spinner="Building candidate pool…")
def build_pool():
    pairs = pd.read_csv(DATA_DIR / "phase2_siamese_training_pairs.csv")
    clean = pd.read_csv(DATA_DIR / "clean_dataset.csv")
    AUDIO = ["Danceability","Energy","Loudness","Speechiness","Acousticness",
             "Instrumentalness","Liveness","Valence","Tempo","Mode","Key"]
    meta = clean[["track_id"] + AUDIO].drop_duplicates("track_id")
    rows = []; seen = set()
    for _, r in pairs.iterrows():
        for side in ["source","target"]:
            tid = r[f"{side}_track_id"]
            if tid in seen: continue
            seen.add(tid)
            rows.append({
                "track_id"  : tid,
                "name"      : r[f"{side}_Track Name"],
                "artist"    : r[f"{side}_Artist Name(s)"],
                "pid"       : r[f"{side}_pid"],
                "fixed_path": r[f"{side}_fixed_matrix_path"],
                "raw_path"  : r[f"{side}_matrix_path"],
                "audio_path": r[f"{side}_audio_path"],
            })
    pool = pd.DataFrame(rows).drop_duplicates("track_id").reset_index(drop=True)
    pool = pool.merge(meta, on="track_id", how="left")
    return pool

@st.cache_data(show_spinner="Embedding candidate pool with Phase 2 encoder…")
def embed_pool(_encoder, fixed_paths_tuple):
    fixed_paths = list(fixed_paths_tuple)
    BATCH = 64
    out = np.zeros((len(fixed_paths), 128), dtype=np.float32)
    buf, idx = [], []
    for i, p in enumerate(fixed_paths):
        buf.append(np.load(p)); idx.append(i)
        if len(buf) == BATCH:
            out[idx] = _encoder.predict(np.stack(buf), verbose=0)
            buf, idx = [], []
    if buf:
        out[idx] = _encoder.predict(np.stack(buf), verbose=0)
    return out

@st.cache_data(show_spinner="Pre-computing Phase-1 weighted vectors…")
def build_weighted(_pool_df, P1_W_VEC):
    P1 = ["Danceability","Energy","Loudness","Speechiness","Acousticness",
          "Instrumentalness","Liveness","Valence","Tempo","Mode","Key"]
    table = {0:0,7:1,2:2,9:3,4:4,11:5,6:6,1:7,8:8,3:9,10:10,5:11}
    raw = _pool_df[P1].astype(float).values.copy()
    raw[:,10] = np.array([table.get(int(k),0) if pd.notna(k) and k!=-1 else 0 for k in raw[:,10]])
    mu = raw.mean(0); sd = raw.std(0) + EPS
    norm = (raw - mu) / sd
    w_feats = norm * np.sqrt(P1_W_VEC)
    w_unit = w_feats / (np.linalg.norm(w_feats, axis=1, keepdims=True) + EPS)
    return raw, w_unit

# --------------------------------------------------------------------------------------
#  Boundary feature extraction (mirrors the notebook)
# --------------------------------------------------------------------------------------
RMS_COL, CENT_COL = 0, 1
MFCC_COLS = slice(5, 18)
CHROMA_COLS = slice(18, 30)
BOUNDARY_FEATURES = [
    "loudness_gap","abs_loudness_gap","is_gain_shock","tail_slope","head_slope",
    "slope_clash","attack_strength_b","decay_completion_a","timbre_l2","timbre_cos",
    "brightness_gap","texture_var_diff","chroma_continuity","boundary_silence_match",
]

def db(x):    return 20.0 * np.log10(np.maximum(x, EPS))
def slope(x): return float(np.polyfit(np.arange(len(x)), x, 1)[0]) if len(x) >= 2 else 0.0
def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na > EPS and nb > EPS else 0.0

def boundary_features(mat_a, mat_b, K=K_WINDOW):
    Ka, Kb = min(K, mat_a.shape[0]), min(K, mat_b.shape[0])
    tail, head = mat_a[-Ka:], mat_b[:Kb]
    tail_db, head_db = db(tail[:, RMS_COL]), db(head[:, RMS_COL])
    end_db, start_db = float(np.mean(tail_db[-3:])), float(np.mean(head_db[:3]))
    loudness_gap = start_db - end_db
    tail_slope, head_slope = slope(tail_db[-4:]), slope(head_db[:4])
    tail_mfcc, head_mfcc = tail[:, MFCC_COLS].mean(0), head[:, MFCC_COLS].mean(0)
    tail_b = float(np.log1p(np.mean(tail[:, CENT_COL])))
    head_b = float(np.log1p(np.mean(head[:, CENT_COL])))
    tail_chr, head_chr = tail[:, CHROMA_COLS].mean(0), head[:, CHROMA_COLS].mean(0)
    return {
        "loudness_gap": loudness_gap, "abs_loudness_gap": abs(loudness_gap),
        "is_gain_shock": float(loudness_gap > 6.0),
        "tail_slope": tail_slope, "head_slope": head_slope,
        "slope_clash": -tail_slope * head_slope,
        "attack_strength_b": float(np.max(head_db[:3]) - np.min(head_db[:3])),
        "decay_completion_a": float(np.max(tail_db) - end_db) / (abs(np.max(tail_db)) + EPS),
        "timbre_l2": float(np.linalg.norm(tail_mfcc - head_mfcc)),
        "timbre_cos": 1.0 - cosine(tail_mfcc, head_mfcc),
        "brightness_gap": abs(head_b - tail_b),
        "texture_var_diff": abs(float(tail[:, MFCC_COLS].var(0).sum()) - float(head[:, MFCC_COLS].var(0).sum())),
        "chroma_continuity": cosine(tail_chr, head_chr),
        "boundary_silence_match": float(end_db < -25 and start_db < -25),
    }

# Raw matrix cache (lazy)
_raw_cache: dict[str, np.ndarray] = {}
def load_raw(path):
    if path not in _raw_cache:
        _raw_cache[path] = np.load(path)
    return _raw_cache[path]

# --------------------------------------------------------------------------------------
#  Scoring helpers (vectorized)
# --------------------------------------------------------------------------------------
def fast_p1_cosine(prev_idx, cand_idx, w_unit):
    return w_unit[cand_idx] @ w_unit[prev_idx]

def full_p1_batch(prev_idx, cand_idx, raw_feats, p1_model):
    sf = raw_feats[prev_idx]; cf = raw_feats[cand_idx]
    diff = np.abs(cf - sf)
    diff[:, 10] = np.minimum(np.abs(sf[10] - cf[:, 10]), 12 - np.abs(sf[10] - cf[:, 10]))
    return p1_model.predict_proba(diff)[:, 1]

def p2_score_batch(prev_idx, cand_idx, pool_emb):
    se = pool_emb[prev_idx]; ce = pool_emb[cand_idx]
    return 1.0 - np.clip(np.linalg.norm(ce - se, axis=1) / 2.0, 0, 1)

def p3_score_batch(prev_idx, cand_idx, pool_df, calib):
    ma = load_raw(pool_df.loc[prev_idx, "raw_path"])
    feats = np.zeros((len(cand_idx), len(BOUNDARY_FEATURES)), dtype=np.float32)
    for i, ci in enumerate(cand_idx):
        bf = boundary_features(ma, load_raw(pool_df.loc[int(ci), "raw_path"]))
        feats[i] = [bf[k] for k in BOUNDARY_FEATURES]
    return calib.predict_proba(feats)[:, 1]

def anchor_similarity(seed_idx, cand_idx, w_unit):
    return (w_unit[cand_idx] @ w_unit[seed_idx] + 1.0) / 2.0

def drift_penalty(seed_idx, cand_idx, raw_feats, tol=0.20):
    sf = raw_feats[seed_idx]; cf = raw_feats[cand_idx]
    abs_floor = np.array([0.05]*8 + [5.0, 1.0, 1.5])
    band = np.maximum(np.abs(sf) * tol, abs_floor)
    soft = band; hard = 2.0 * band
    diff = np.abs(cf - sf)
    pen = np.where(diff <= soft, 0.0,
            np.where(diff <= hard, (diff - soft) / (hard - soft + EPS), 1.0))
    return pen.mean(axis=1)

# --------------------------------------------------------------------------------------
#  Beam search
# --------------------------------------------------------------------------------------
def generate_queue(seed_track_id, length, beam_k, top_n, alpha, beta, gamma,
                   pool_df, raw_feats, w_unit, pool_emb, p1_model, calib, stack, lam=0.30):
    seed_idx = int(pool_df.index[pool_df["track_id"] == seed_track_id][0])
    beams = [{"path":[seed_idx], "trans":[], "finals":[], "p1s":[], "p2s":[], "p3s":[], "cum":1.0}]
    universe = np.arange(len(pool_df))

    for _ in range(length):
        cand_set = []
        for b in beams:
            last = b["path"][-1]; used = set(b["path"])
            mask = np.array([i not in used for i in universe])
            cand_pool = universe[mask]
            cos = fast_p1_cosine(last, cand_pool, w_unit)
            top = cand_pool[np.argsort(-cos)[:top_n]]
            drift = drift_penalty(seed_idx, top, raw_feats)
            keep = drift < 1.0
            top, drift = top[keep], drift[keep]
            if len(top) == 0: continue
            p1 = full_p1_batch(last, top, raw_feats, p1_model)
            p2 = p2_score_batch(last, top, pool_emb)
            p3 = p3_score_batch(last, top, pool_df, calib)
            trans = stack.predict_proba(np.column_stack([p1, p2, p3]))[:, 1]
            anch = anchor_similarity(seed_idx, top, w_unit)
            final = alpha * trans + beta * anch + gamma * (1.0 - drift)
            for i, ci in enumerate(top):
                new_t = b["trans"] + [float(trans[i])]
                new_f = b["finals"] + [float(final[i])]
                cum = float(np.mean(new_f) - lam * np.std(new_f)) if len(new_f) > 1 else float(final[i])
                cand_set.append({
                    "path"  : b["path"]  + [int(ci)],
                    "trans" : new_t, "finals": new_f,
                    "p1s"   : b["p1s"]   + [float(p1[i])],
                    "p2s"   : b["p2s"]   + [float(p2[i])],
                    "p3s"   : b["p3s"]   + [float(p3[i])],
                    "cum"   : cum,
                })
        if not cand_set: break
        beams = sorted(cand_set, key=lambda x: -x["cum"])[:beam_k]
    return beams[0]

# --------------------------------------------------------------------------------------
#  UI
# --------------------------------------------------------------------------------------
st.markdown("# 🎧 Flow-Aware Music")
st.markdown("##### Vibe-preserving queue generator · pick a seed song, get a smooth queue")

# Boot
p1_model, P1_W_VEC = load_p1()
encoder            = load_p2_encoder()
calib, stack, meta = load_p3()
pool_df            = build_pool()
raw_feats, w_unit  = build_weighted(pool_df, P1_W_VEC)
pool_emb           = embed_pool(encoder, tuple(pool_df["fixed_path"].tolist()))

# Sidebar — controls
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    queue_length = st.slider("Queue length", 5, 20, 15)
    beam_k       = st.slider("Beam width (k)", 1, 5, 3)
    top_n        = st.slider("Top-N candidates", 20, 150, 60, step=10)
    st.markdown("---")
    st.markdown("### 🎚️ Score weights")
    alpha = st.slider("α  (transition smoothness)", 0.0, 1.0, 0.40, 0.05)
    beta  = st.slider("β  (anchor to seed)",       0.0, 1.0, 0.35, 0.05)
    gamma = st.slider("γ  (drift bound)",          0.0, 1.0, 0.25, 0.05)
    st.markdown("---")
    st.caption(f"Pool size: **{len(pool_df)}** tracks")
    st.caption(f"Stacking τ: **{meta['thresholds']['tau_stack']:.3f}**")

# Seed selection
st.markdown("### 1. Choose a seed song")
labels = [f"{r['name']} — {r['artist']}" for _, r in pool_df.iterrows()]
choice = st.selectbox("Search by name or artist", options=labels, index=0)
seed_idx = labels.index(choice)
seed_row = pool_df.iloc[seed_idx]

# Seed card
st.markdown(
    f"""
    <div class="seed-card">
      <div style="font-size:13px;color:#aaa;text-transform:uppercase;letter-spacing:1px;">SEED</div>
      <div style="font-size:24px;font-weight:700;color:#fff;margin-top:4px;">{seed_row['name']}</div>
      <div style="font-size:15px;color:#ccc;margin-top:2px;">{seed_row['artist']}</div>
      <div style="font-size:12px;color:#888;margin-top:8px;">
        Tempo {seed_row['Tempo']:.0f} · Energy {seed_row['Energy']:.2f} ·
        Loudness {seed_row['Loudness']:.1f} dB · Valence {seed_row['Valence']:.2f}
      </div>
    </div>
    """, unsafe_allow_html=True
)

# Optional audio preview of the seed
seed_audio = seed_row.get("audio_path")
if isinstance(seed_audio, str) and Path(seed_audio).exists():
    st.audio(seed_audio)

st.markdown("### 2. Generate the queue")
go = st.button("🎵  Generate Queue", type="primary", use_container_width=True)

if go:
    with st.spinner("Running beam search… this may take 10–30 seconds on first run while raw matrices cache."):
        result = generate_queue(
            seed_row["track_id"], queue_length, beam_k, top_n,
            alpha, beta, gamma,
            pool_df, raw_feats, w_unit, pool_emb, p1_model, calib, stack
        )

    queue = pool_df.iloc[result["path"]].reset_index(drop=True)
    queue["transition"] = [None] + [round(s, 3) for s in result["trans"]]
    queue["final"]      = [None] + [round(s, 3) for s in result["finals"]]
    queue["p1"]         = [None] + [round(s, 3) for s in result["p1s"]]
    queue["p2"]         = [None] + [round(s, 3) for s in result["p2s"]]
    queue["p3"]         = [None] + [round(s, 3) for s in result["p3s"]]

    drift = drift_penalty(seed_idx, np.array(result["path"]), raw_feats)
    mean_t = float(np.mean(result["trans"]))
    std_t  = float(np.std(result["trans"]))

    # Headline metrics
    st.markdown("### 3. Queue summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean transition", f"{mean_t:.3f}", help="Higher is smoother (target ≥ 0.78)")
    c2.metric("Transition std",  f"{std_t:.3f}",  help="Lower is more uniform (target ≤ 0.08)")
    c3.metric("Mean drift",      f"{drift.mean():.3f}", help="Lower stays closer to seed (target ≤ 0.25)")
    c4.metric("Max drift",       f"{drift.max():.3f}",  help="Worst position drift")

    # Plots
    st.markdown("### 4. Queue dynamics")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
    ax[0].plot(drift, marker="o", color="#4f8ef7"); ax[0].set_ylim(0, 1)
    ax[0].axhline(0.25, color="#888", linestyle="--", alpha=0.6)
    ax[0].set_title("Anchor drift across queue", color="white")
    ax[0].set_xlabel("Position"); ax[0].set_ylabel("Drift penalty"); ax[0].grid(True, alpha=0.3)
    ax[1].plot(result["trans"], marker="o", color="#1aaf5d"); ax[1].set_ylim(0, 1)
    ax[1].axhline(0.78, color="#888", linestyle="--", alpha=0.6)
    ax[1].set_title("Transition score per step", color="white")
    ax[1].set_xlabel("Step"); ax[1].set_ylabel("Stacked score"); ax[1].grid(True, alpha=0.3)
    fig.patch.set_alpha(0)
    for a in ax: a.set_facecolor("#1a1a2e")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Queue cards
    st.markdown("### 5. Your queue")
    for i, row in queue.iterrows():
        score = row["transition"]
        if i == 0:
            pill_html = '<span class="score-pill">SEED</span>'
        else:
            cls = "score-pill" if score >= 0.85 else ("score-pill score-mid" if score >= 0.7 else "score-pill score-low")
            pill_html = f'<span class="{cls}">{score:.3f}</span>'
        card = f"""
        <div class="queue-card">
            <div class="queue-num">#{i:02d}</div>
            <div class="queue-title">{row['name']} {pill_html}</div>
            <div class="queue-artist">{row['artist']}</div>
        </div>
        """
        st.markdown(card, unsafe_allow_html=True)
        ap = row.get("audio_path")
        if isinstance(ap, str) and Path(ap).exists():
            with st.expander("▶  Play audio", expanded=False):
                st.audio(ap)
                with st.expander("Component scores", expanded=False):
                    st.write({
                        "Phase 1 (global)"  : row["p1"],
                        "Phase 2 (envelope)": row["p2"],
                        "Phase 3 (boundary)": row["p3"],
                        "Final (α·trans + β·anchor + γ·(1-drift))": row["final"],
                    })

    st.markdown(
        """
        <div class="footer-note">
        α·transition_score + β·anchor_similarity + γ·(1 − drift_penalty) · beam-search ranks queues by mean − λ·std
        </div>
        """, unsafe_allow_html=True
    )
else:
    st.info("Tune settings in the sidebar, then click **Generate Queue**.")
