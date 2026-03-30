import streamlit as st
import pandas as pd
import joblib
import numpy as np


try:
    model = joblib.load('house_price_model.pkl')
except:
    st.error("Model file not found! Please run your training script first to generate 'house_price_model.pkl'.")

st.set_page_config(page_title="Ames House Price Predictor", page_icon="🏡")

st.title("🏡 Ames Housing Price Predictor")
st.markdown("Enter the property details below to get an AI-powered valuation.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Structure & Size")
    qual = st.slider("Overall House Quality (1-10)", 1, 10, 6)
    total_sf = st.number_input("Total Square Footage (Incl. Bsmt)", 500, 10000, 2000)
    liv_area = st.number_input("Above Ground Living Area (sqft)", 500, 5000, 1500)
    age = st.number_input("Age of the House (Years)", 0, 150, 20)

with col2:
    st.subheader("Location & Amenities")
    nb = st.selectbox("Neighborhood", ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'SawyerW'])
    garage = st.selectbox("Garage Capacity (Cars)", [0, 1, 2, 3, 4])
    bath = st.number_input("Total Bathrooms", 1.0, 6.0, 2.0)
    ac = st.selectbox("Central Air Conditioning", ['Y', 'N'])

input_data = pd.DataFrame({
    'Overall Qual': [qual], 
    'Gr Liv Area': [liv_area], 
    'TotalSF': [total_sf],
    'TotalBathrooms': [bath], 
    'HouseAge': [age], 
    'Garage Cars': [garage],
    'Garage Area': [garage * 240], # Rough estimate: 240 sqft per car
    'Full Bath': [int(bath)], 
    'Fireplaces': [1],
    'MS Zoning': ['RL'], # Defaulting to Residential Low density
    'Neighborhood': [nb], 
    'Kitchen Qual': ['Gd'], # Defaulting to Good
    'Exter Qual': ['Gd'],   # Defaulting to Good
    'Central Air': [ac]
})

if st.button("Predict Market Value"):
    log_prediction = model.predict(input_data)
    # We used log1p in training, so we use expm1 to get the actual dollar amount
    final_price = np.expm1(log_prediction)[0]
    st.success(f"### Estimated Price: ${final_price:,.2f}")
    st.balloons()


import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Price Predictor", "Data Visualizations"])

if page == "Price Predictor":
    st.write("Use the sliders to predict house prices.")

elif page == "Data Visualizations":
    st.header("📊 Data Insights & Model Performance")
    
    train_df = pd.read_csv('train.csv')
    
    st.subheader("1. Distribution of House Prices")
    fig, ax = plt.subplots()
    sns.histplot(train_df['SalePrice'], kde=True, ax=ax, color='blue')
    plt.title("Price Distribution (Raw Data)")
    st.pyplot(fig)
    st.write("Insight: Notice the 'Right Skew'. This is why we used Log-Transformation in our model.")

    st.subheader("2. Quality vs. Market Value")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Overall Qual', y='SalePrice', data=train_df, ax=ax2)
    plt.title("Overall Quality vs Sale Price")
    st.pyplot(fig2)
    st.write("Insight: Price increases exponentially as Quality moves from 6 to 10.")

    st.subheader("3. Feature Importance (Correlation)")
    numeric_df = train_df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()['SalePrice'].sort_values(ascending=False).head(10)
    fig3, ax3 = plt.subplots()
    corr.plot(kind='bar', ax=ax3, color='teal')
    plt.title("Top 10 Factors Driving Price")
    st.pyplot(fig3)
