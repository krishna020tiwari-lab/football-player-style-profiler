# 🧩 Football Player Style Profiler (Unsupervised Learning)

An Unsupervised Machine Learning project that discovers distinct tactical profiles of European football players using **K-Means Clustering** and statistical performance metrics.

---

## 📌 Project Overview

Unlike traditional position classification, this project removes pre-defined position labels (`DF`, `MF`, `FW`) and groups players based purely on their on-field output (e.g., Progressive Passes, Goals, Expected Assists, Progressive Carries).

* **Machine Learning Type:** Unsupervised Learning (Clustering)
* **Algorithm:** K-Means Clustering (`scikit-learn`)
* **Feature Scaling:** `StandardScaler`
* **Dataset:** Top 5 European League Player Stats (2024–2025)
* **Interactive Dashboard:** Streamlit

---

## 💡 Key Machine Learning Concepts Applied

1. **Unsupervised Clustering:** Partitioning unlabelled player statistical features into distinct tactical clusters ($K=4$).
2. **Standardization (`StandardScaler`):** Rescaling features to equal variance ($mean=0, std=1$) so high-range stats (e.g., Passes) do not dominate low-range stats (e.g., Goals) during Euclidean distance calculation.
3. **Cluster Profiling:** Analyzing mean feature vectors per cluster to assign tactical profiles (e.g., *Clinical Goalscorers*, *Deep-Lying Playmakers*, *Dribblers/Carriers*).

---

## 🛠️ Project Structure

```text
├── train_clusters.py      # K-Means training & feature scaling pipeline
├── app_clusters.py        # Streamlit interactive cluster explorer
├── player_clusters.pkl    # Serialized KMeans model & scaler payload
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
