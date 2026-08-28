import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("players-data-2024-2025.csv")

df_filtered = df[df['Min'] >= 450].copy()

style_features = ['Gls', 'Sh', 'SoT', 'Ast', 'xG', 'xAG', 'PrgP', 'PrgC']
X_raw = df_filtered[style_features].fillna(0)

scaler = StandardScaler()
X_Scaled = scaler.fit_transform(X_raw)

kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X_Scaled)
df_filtered['Cluster'] = kmeans.labels_

cluster_profiles = df_filtered.groupby('Cluster')[style_features].mean()
print("📊 Cluster Feature Averages:")
print(cluster_profiles)

joblib.dump(
    {
        'kmeans': kmeans,
        'scaler': scaler,
        'features': style_features,
    },
    'player_clusters.pkl',
)
print("💾 Exported player_clusters.pkl successfully!")

