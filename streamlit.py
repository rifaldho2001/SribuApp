import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load scaler, PCA, and clustering model
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
clustering_model = joblib.load('kmeans_model.pkl')

# Title and description
st.title("Clustering Customer Sribu")
st.write("Masukkan nilai fitur untuk mengetahui cluster dari pelanggan.")

# Feature input form
st.sidebar.header("Input Features")
recency = st.sidebar.number_input("Recency (days)", min_value=0, value=10, step=1)
frequency = st.sidebar.number_input("Frequency (count)", min_value=0, value=1, step=1)
monetary = st.sidebar.number_input("Monetary (total paid)", min_value=0.0, value=100.0, step=0.1)

# Category inputs (one-hot encoded categories)
st.sidebar.subheader("Category Features")
categories = {
    "Desain Grafis & Branding": st.sidebar.number_input("Desain Grafis & Branding", min_value=0, value=0, step=1),
    "Gaya Hidup": st.sidebar.number_input("Gaya Hidup", min_value=0, value=0, step=1),
    "Konsultasi": st.sidebar.number_input("Konsultasi", min_value=0, value=0, step=1),
    "Pemasaran & Periklanan": st.sidebar.number_input("Pemasaran & Periklanan", min_value=0, value=0, step=1),
    "Penulisan & Penerjemahan": st.sidebar.number_input("Penulisan & Penerjemahan", min_value=0, value=0, step=1),
    "Video, Fotografi & Audio": st.sidebar.number_input("Video, Fotografi & Audio", min_value=0, value=0, step=1),
    "Web & Pemrograman": st.sidebar.number_input("Web & Pemrograman", min_value=0, value=0, step=1),
}

# Collect inputs into a dataframe
input_features = pd.DataFrame([[
    recency, frequency, monetary,
    categories["Desain Grafis & Branding"],
    categories["Gaya Hidup"],
    categories["Konsultasi"],
    categories["Pemasaran & Periklanan"],
    categories["Penulisan & Penerjemahan"],
    categories["Video, Fotografi & Audio"],
    categories["Web & Pemrograman"]
]], columns=[
    "recency", "frequency", "monetary",
    "category_Desain Grafis & Branding",
    "category_Gaya Hidup",
    "category_Konsultasi",
    "category_Pemasaran & Periklanan",
    "category_Penulisan & Penerjemahan",
    "category_Video, Fotografi & Audio",
    "category_Web & Pemrograman"
])

# Tombol Generate
if st.sidebar.button("Generate"):
    # Scaling the input features
    scaled_features = scaler.transform(input_features)

    # Reducing dimensions using PCA
    pca_features = pca.transform(scaled_features)

    # Predicting the cluster
    predicted_cluster = clustering_model.predict(pca_features)[0]

    # Display results
    st.subheader("Cluster Prediction")
    st.write(f"Hasil prediksi: Anda termasuk dalam Cluster {predicted_cluster}")