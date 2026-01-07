import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Page title
st.title("DBSCAN Clustering App")

# Load dataset
df = pd.read_csv("wine_clustering_data.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Sidebar parameters
st.sidebar.header("DBSCAN Parameters")
eps = st.sidebar.slider("EPS value", 0.1, 5.0, 1.5)
min_samples = st.sidebar.slider("Min Samples", 2, 10, 5)

# DBSCAN model
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters = dbscan.fit_predict(X_scaled)

df["Cluster"] = clusters

st.subheader("Cluster Counts")
st.write(df["Cluster"].value_counts())

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# Plot
fig, ax = plt.subplots()
scatter = ax.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"])
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_title("DBSCAN Clustering Output")

st.pyplot(fig)

# Outliers
outliers = df[df["Cluster"] == -1]
st.subheader("Outliers")
st.write(outliers)
