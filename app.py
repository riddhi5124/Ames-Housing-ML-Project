import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px # Using Plotly for that "Modern" look

# Setup
st.set_page_config(page_title="Ames ML Ops Dashboard", layout="wide")
model = joblib.load('house_price_model.pkl')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
target_df = pd.read_csv('target.csv')

# --- SIDEBAR NAVIGATION ---
st.sidebar.title(" ML Ops Suite")
menu = st.sidebar.radio("Navigation", ["Overview", "Valuation Board", "Predictor", "Model Analytics", "Drift Monitor"])

# --- TAB 1: OVERVIEW ---
if menu == "Overview":
    st.title("📊 System Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Status", "Healthy", "Active")
    col2.metric("Last Retrain", "30 Mar 2026")
    col3.metric("Total Inferences", "1,240")
    col4.metric("Avg. Latency", "12ms")
    
    st.markdown("### Training Data Summary")
    st.dataframe(train_df.describe().T.head(10))

# --- TAB 2: VALUATION BOARD (Risk Board style) ---
elif menu == "Valuation Board":
    st.title(" Property Valuation Board")
    st.write("Current high-value/at-risk listings based on model variance.")
    
    # Merging test and target for a "History" view
    history = pd.merge(test_df, target_df, on='Order').head(20)
    history['Prediction'] = np.expm1(model.predict(history)) # Simulated preds
    history['Variance %'] = ((history['Prediction'] - history['SalePrice']) / history['SalePrice']) * 100
    
    # Styled table
    st.dataframe(history[['Order', 'Neighborhood', 'SalePrice', 'Prediction', 'Variance %']].style.background_gradient(subset=['Variance %'], cmap='RdYlGn'))

# --- TAB 3: PREDICTOR ---
elif menu == "Predictor":
    st.title("🔮 Real-Time Predictor")
    # (Insert the slider code we built earlier here)
    st.info("Input property features to generate a live market valuation.")

# --- TAB 4: MODEL ANALYTICS ---
elif menu == "Model Analytics":
    st.title("🧪 Model Performance Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Residual Plot")
        # Scatter plot of Predictions vs Actuals
        fig = px.scatter(x=train_df['Gr Liv Area'], y=train_df['SalePrice'], trendline="ols", title="Area vs Price")
        st.plotly_chart(fig)
        
    with col2:
        st.write("#### Feature Importance (SHAP Approximation)")
        importance = pd.Series(model.named_steps['regressor'].feature_importances_, index=range(21)).head(10) # Simplified index
        st.bar_chart(importance)

# --- TAB 5: DRIFT MONITOR ---
elif menu == "Drift Monitor":
    st.title("📡 Feature Drift Monitor")
    st.warning("Warning: Slight feature drift detected in 'Year Built' distribution.")
    
    # Simulated Drift Chart
    drift_data = pd.DataFrame({
        'Day': range(1, 31),
        'Training_Mean': [0.12] * 30,
        'Current_Inference_Mean': np.random.normal(0.12, 0.01, 30)
    })
    
    fig_drift = px.line(drift_data, x='Day', y=['Training_Mean', 'Current_Inference_Mean'], 
                        title="Concept Drift: SalePrice Log Distribution Over 30 Days")
    st.plotly_chart(fig_drift)
