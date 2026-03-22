import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Page Config
st.set_page_config(page_title="Telecom Intelligent Systems", layout="wide")

@st.cache_data
def get_data():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

df = get_data()

# --- SIDEBAR ---
st.sidebar.header("Customer Input")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 120, 70)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# --- MAIN PAGE ---
st.title("📡 Telecom Customer Intelligence Platform")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Unsupervised Segmentation")
    # Quick K-Means for the demo
    km_features = df[['tenure', 'MonthlyCharges']]
    kmeans = KMeans(n_clusters=3, n_init=10).fit(km_features)
    df['Cluster'] = kmeans.labels_
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='Cluster', palette='Set1', ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("2. Supervised Churn Prediction")
    # Simplified training for the demo app
    le = LabelEncoder()
    df['Churn_Label'] = le.fit_transform(df['Churn'])
    
    X = df[['tenure', 'MonthlyCharges']]
    y = df['Churn_Label']
    
    model = RandomForestClassifier(n_estimators=100).fit(X, y)
    
    # Prediction logic
    input_data = np.array([[tenure, monthly_charges]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if st.button("Analyze Customer"):
        if prediction[0] == 1:
            st.error(f"HIGH CHURN RISK: {probability:.1%}")
            st.write("Recommendation: Offer a long-term contract discount.")
        else:
            st.success(f"LOW CHURN RISK: {probability:.1%}")
            st.write("Recommendation: Target for premium service upsell.")