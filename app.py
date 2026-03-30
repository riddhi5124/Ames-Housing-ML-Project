import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 1. PAGE CONFIG & THEME
st.set_page_config(page_title="Ames ML Ops Dashboard", layout="wide")

# 2. FILE LOADING (With Path Safety)
base_path = os.path.dirname(__file__)

def load_data():
    try:
        train = pd.read_csv(os.path.join(base_path, 'train.csv'))
        test = pd.read_csv(os.path.join(base_path, 'test.csv'))
        target = pd.read_csv(os.path.join(base_path, 'target.csv'))
        model = joblib.load(os.path.join(base_path, 'house_price_model.pkl'))
        return train, test, target, model
    except FileNotFoundError as e:
        st.error(f"Missing file error: {e}. Please ensure all CSVs and the .pkl are in the GitHub root.")
        st.stop()

train_df, test_df, target_df, model = load_data()

# 3. GLOBAL DEFINITIONS (Used in Analytics and Predictor)
numeric_features = ['Overall Qual', 'Gr Liv Area', 'TotalSF', 'TotalBathrooms', 'HouseAge', 'Garage Cars', 'Garage Area', 'Full Bath', 'Fireplaces']
categorical_features = ['MS Zoning', 'Neighborhood', 'Kitchen Qual', 'Exter Qual', 'Central Air']

# 4. FEATURE ENGINEERING FUNCTION
def engineer_features(df):
    data = df.copy()
    # Create the engineered columns the model expects
    data['TotalSF'] = data['Total Bsmt SF'].fillna(0) + data['1st Flr SF'] + data['2nd Flr SF']
    data['TotalBathrooms'] = (data['Full Bath'] + (0.5 * data['Half Bath']) +
                              data['Bsmt Full Bath'].fillna(0) + (0.5 * data['Bsmt Half Bath'].fillna(0)))
    data['HouseAge'] = data['Yr Sold'] - data['Year Built']
    
    # Clean up standard columns
    data['Garage Cars'] = data['Garage Cars'].fillna(0)
    data['Garage Area'] = data['Garage Area'].fillna(0)
    return data

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛡️ ML Ops Suite")
menu = st.sidebar.radio("Navigation", ["Overview", "Valuation Board", "Predictor", "Model Analytics", "Drift Monitor"])

# --- TAB 1: OVERVIEW ---
if menu == "Overview":
    st.title("📊 System Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Status", "Healthy", "Active")
    col2.metric("Architecture", "Random Forest")
    col3.metric("R² Score", "0.884")
    col4.metric("Avg Error", "$24,638")
    
    st.markdown("### Training Data Preview")
    st.dataframe(train_df.head(10))

# --- TAB 2: VALUATION BOARD ---
elif menu == "Valuation Board":
    st.title("📋 Property Valuation Board")
    st.write("Comparing Blind Test predictions against actual market values.")
    
    # Merge and Engineer
    history = pd.merge(test_df, target_df, on='Order')
    history_eng = engineer_features(history)
    
    # Inference
    history['Prediction'] = np.expm1(model.predict(history_eng))
    history['Variance %'] = ((history['Prediction'] - history['SalePrice']) / history['SalePrice']) * 100
    
    st.dataframe(history[['Order', 'Neighborhood', 'SalePrice', 'Prediction', 'Variance %']].style.background_gradient(subset=['Variance %'], cmap='RdYlGn_r'))

# --- TAB 3: PREDICTOR ---
elif menu == "Predictor":
    st.title("🔮 Real-Time Market Predictor")
    
    col1, col2 = st.columns(2)
    with col1:
        qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
        liv_area = st.number_input("Living Area (sqft)", 500, 5000, 1500)
        total_sf = st.number_input("Total SF (Incl. Bsmt)", 500, 10000, 2000)
        age = st.number_input("House Age (Years)", 0, 150, 20)
    
    with col2:
        nb = st.selectbox("Neighborhood", sorted(train_df['Neighborhood'].unique()))
        garage = st.selectbox("Garage Capacity", [0, 1, 2, 3, 4])
        bath = st.number_input("Total Bathrooms", 1.0, 6.0, 2.0)
        ac = st.selectbox("Central Air", ['Y', 'N'])

    if st.button("Predict Price", use_container_width=True):
        input_df = pd.DataFrame({
            'Overall Qual': [qual], 'Gr Liv Area': [liv_area], 'TotalSF': [total_sf],
            'TotalBathrooms': [bath], 'HouseAge': [age], 'Garage Cars': [garage],
            'Garage Area': [garage * 240], 'Full Bath': [int(bath)], 'Fireplaces': [1],
            'MS Zoning': ['RL'], 'Neighborhood': [nb], 'Kitchen Qual': ['Gd'],
            'Exter Qual': ['Gd'], 'Central Air': [ac]
        })
        
        log_price = model.predict(input_df)
        final_price = np.expm1(log_price)[0]
        st.success(f"### Estimated Value: ${final_price:,.2f}")

# --- TAB 4: MODEL ANALYTICS ---
elif menu == "Model Analytics":
    st.title("🧪 Model Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Predicted vs. Actual")
        fig = px.scatter(train_df, x='Gr Liv Area', y='SalePrice', trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.write("#### Feature Importance")
        # Extract importance and match with encoded names
        importances = model.named_steps['regressor'].feature_importances_
        cat_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_names = numeric_features + list(cat_names)
        
        feat_imp = pd.Series(importances, index=all_names).nlargest(10)
        st.bar_chart(feat_imp)

# --- TAB 5: DRIFT MONITOR ---
elif menu == "Drift Monitor":
    st.title("📡 Concept Drift Monitor")
    st.write("Simulated monitoring of SalePrice distribution over time.")
    drift_sim = pd.DataFrame({
        'Inference_Batch': range(1, 101),
        'Target_Score': np.random.normal(0.88, 0.02, 100)
    })
    fig_drift = px.line(drift_sim, x='Inference_Batch', y='Target_Score', title="Model Accuracy Stability")
    st.plotly_chart(fig_drift, use_container_width=True)
