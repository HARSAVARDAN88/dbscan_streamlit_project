import streamlit as st
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

st.title("DBSCAN Clustering App")

# Load dataset
df = pd.read_csv("dataset.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Sidebar inputs
st.sidebar.header("DBSCAN Parameters")
eps = st.sidebar.slider("EPS", 0.1, 5.0, 1.5)
min_samples = st.sidebar.slider("Min Samples", 2, 10, 5)

# DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters = dbscan.fit_predict(X_scaled)

df["Cluster"] = clusters

st.subheader("Cluster Count")
st.write(df["Cluster"].value_counts())

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# Plotly scatter plot
fig = px.scatter(
    df,
    x="PCA1",
    y="PCA2",
    color="Cluster",
    title="DBSCAN Clustering Output"
)

st.plotly_chart(fig)

# Show outliers
st.subheader("Outliers (Cluster = -1)")
st.write(df[df["Cluster"] == -1])
