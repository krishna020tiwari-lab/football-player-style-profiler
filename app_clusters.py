import joblib
import numpy as np
import pandas as pd
import streamlit as st

payload = joblib.load('player_clusters.pkl')
kmeans = payload['kmeans']
scaler = payload['scaler']
features = payload['features']


profile_names = {
    0: 'Deep-Lying Playmakers (High PrgP, Moderate Attacking)',
    1: 'Clinical Goalscorers (High Gls, Sh, xG)',
    2: 'Ball Carriers / Wingers (High PrgC, xAG, Ast)',
    3: 'Defensive / Conservative Players (Low Attacking Output)',
}

st.set_page_config(page_title="Player Style Profiler", page_icon="🧩")
st.title("🧩 Unsupervised Football Player Style Profiler")
st.markdown("Categorize player profiles using **K-Means Clustering**.")

cols = st.columns(2)
inputs = {}

for idx, feat in enumerate(features):
  col = cols[idx % 2]
  inputs[feat] = col.number_input(f"Value for {feat}", min_value=0.0, value=10.0)

if st.button("🔍 Assign Style Cluster", type="primary"):
  raw_vals = np.array([[inputs[f] for f in features]])
  scaled_vals = scaler.transform(raw_vals)
  cluster_id = int(kmeans.predict(scaled_vals)[0])

  st.markdown("---")
  st.subheader(f"Assigned Cluster: **Cluster {cluster_id}**")
  st.success(f"Profile Type: **{profile_names.get(cluster_id, 'Custom Profile')}**")