import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier # Ensure pip install xgboost
from sklearn.metrics import mean_absolute_error, r2_score, precision_score, recall_score, fbeta_score, roc_auc_score

# Page Configuration
st.set_page_config(page_title="Ames Housing ML Project", layout="wide")

# ---------------------------------------------------------
# DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    target = pd.read_csv('target.csv')
    test_full = pd.merge(test, target, on='Order')
    
    # Combined cleaning logic
    def clean(df):
        df = df.copy().drop(['Order', 'PID'], axis=1, errors='ignore')
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            df[col] = df[col].fillna('None')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        return df

    train_c = clean(train)
    test_c = clean(test_full)
    return train, test_full, train_c, test_c

train_raw, test_raw, train_c, test_c = load_and_clean_data()

# ---------------------------------------------------------
# MODEL TRAINING (THE "ENGINE")
# ---------------------------------------------------------
@st.cache_resource
def train_models(train_df, test_df):
    X_train = train_df.drop('SalePrice', axis=1)
    y_train = train_df['SalePrice']
    X_test = test_df.drop('SalePrice', axis=1)
    y_test = test_df['SalePrice']
    
    # Classification Setup (High Value = 1 if > Median)
    threshold = y_train.median()
    y_train_cls = (y_train > threshold).astype(int)
    y_test_cls = (y_test > threshold).astype(int)
    
    # Scaler for SVR/Linear
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 1. Regression Models
    reg_list = {
        "Linear Regression": LinearRegression().fit(X_train_s, y_train),
        "SVR": SVR(kernel='rbf', C=1e5).fit(X_train_s, y_train),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05).fit(X_train, y_train)
    }

    # 2. Classification Models (for Probability/Metrics)
    clf_list = {
        "Linear Regression": LogisticRegression(max_iter=1000).fit(X_train_s, y_train_cls),
        "SVR": SVC(probability=True).fit(X_train_s, y_train_cls),
        "Random Forest": RandomForestClassifier(random_state=42).fit(X_train, y_train_cls),
        "XGBoost": XGBClassifier(eval_metric='logloss').fit(X_train, y_train_cls)
    }

    # Calculate Real Metrics
    perf_data = []
    for name in reg_list.keys():
        # Reg metrics
        m_reg = reg_list[name]
        p_reg = m_reg.predict(X_test_s if name in ["Linear Regression", "SVR"] else X_test)
        
        # Clf metrics
        m_clf = clf_list[name]
        p_clf = m_clf.predict(X_test_s if name in ["Linear Regression", "SVR"] else X_test)
        prob_clf = m_clf.predict_proba(X_test_s if name in ["Linear Regression", "SVR"] else X_test)[:, 1]
        
        perf_data.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, p_reg),
            "R2": r2_score(y_test, p_reg),
            "AUC": roc_auc_score(y_test_cls, prob_clf),
            "Precision": precision_score(y_test_cls, p_clf),
            "Recall": recall_score(y_test_cls, p_clf),
            "F2": fbeta_score(y_test_cls, p_clf, beta=2)
        })

    return reg_list, clf_list, pd.DataFrame(perf_data), threshold

reg_models, clf_models, metrics_df, price_threshold = train_models(train_c, test_c)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("SML Project: Ames Housing")
menu = st.sidebar.radio("Navigation", ["Dataset Overview", "Feature Importance", "Price Predictor", "Model Analytics"])

# ---------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------
if menu == "Dataset Overview":
    st.title("📊 Ames Housing Overview")
    
    # Row 1: Key Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Listings", len(train_raw) + len(test_raw))
    c2.metric("Neighborhoods", train_raw['Neighborhood'].nunique())
    c3.metric("Avg Lot Size", f"{train_raw['Lot Area'].mean():,.0f} sf")
    c4.metric("Median Price", f"${price_threshold:,.0f}")

    st.divider()

    # Row 2: Visuals
    left, right = st.columns(2)
    with left:
        st.subheader("Neighborhood Price Distribution")
        fig, ax = plt.subplots()
        sns.boxplot(data=train_raw, x='SalePrice', y='Neighborhood', ax=ax, palette='coolwarm')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with right:
        st.subheader("Listing Tiers (Classification)")
        bins = [0, 130000, 210000, np.inf]
        names = ['Low', 'Medium', 'Premium']
        train_raw['Tier'] = pd.cut(train_raw['SalePrice'], bins=bins, labels=names)
        counts = train_raw['Tier'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
        st.pyplot(fig2)

# ---------------------------------------------------------
# PAGE 2: FEATURE IMPORTANCE
# ---------------------------------------------------------
elif menu == "Feature Importance":
    st.title("🎯 Best Feature Extraction")
    
    # Get importance from Random Forest
    rf_model = reg_models["Random Forest"]
    feats = train_c.drop('SalePrice', axis=1).columns
    imp_df = pd.DataFrame({'Feature': feats, 'Score': rf_model.feature_importances_}).sort_values(by='Score', ascending=False)

    st.subheader("Top Predictors Visualized")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=imp_df.head(10), x='Score', y='Feature', palette='viridis', ax=ax)
    st.pyplot(fig)

    st.subheader("Top 10 Rows of Key Data (Filtered by High Price)")
    top_cols = imp_df.head(5)['Feature'].tolist()
    st.dataframe(train_raw[top_cols + ['SalePrice']].sort_values(by='SalePrice', ascending=False).head(10))

# ---------------------------------------------------------
# PAGE 3: PREDICTOR
# ---------------------------------------------------------
elif menu == "Price Predictor":
    st.title("🏠 Live Price Predictor")
    
    # Inputs based on top features
    col1, col2 = st.columns(2)
    with col1:
        qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
        gr_area = st.number_input("Living Area (SqFt)", 500, 5000, 1500)
    with col2:
        year = st.number_input("Year Built", 1880, 2010, 1995)
        garage = st.selectbox("Garage Cars", [0, 1, 2, 3, 4])

    # Simple logic to create a dummy input row for prediction
    # We use the median of other features to make a full row for the model
    input_data = train_c.drop('SalePrice', axis=1).median().values.reshape(1, -1)
    input_df = pd.DataFrame(input_data, columns=train_c.drop('SalePrice', axis=1).columns)
    
    # Update with user inputs
    input_df['Overall Qual'] = qual
    input_df['Gr Liv Area'] = gr_area
    input_df['Year Built'] = year
    input_df['Garage Cars'] = garage

    # Predict using XGBoost (the best model)
    price_pred = reg_models["XGBoost"].predict(input_df)[0]
    prob_pred = clf_models["XGBoost"].predict_proba(input_df)[0][1]

    st.success(f"### Predicted Market Value: ${price_pred:,.2f}")
    st.write(f"**Chances of this being a 'Premium Sale' (Above ${price_threshold:,.0f}):**")
    st.progress(float(prob_pred))
    st.write(f"{prob_pred*100:.1f}% Probability")

# ---------------------------------------------------------
# PAGE 4: MODEL ANALYTICS
# ---------------------------------------------------------
elif menu == "Model Analytics":
    st.title("📈 Performance Analysis & Comparison")
    
    st.subheader("Real-time Model Metrics")
    st.table(metrics_df.style.highlight_max(axis=0, subset=['R2', 'AUC', 'F2'], color='lightgreen'))

    st.subheader("Model Comparison: AUC vs F2 Score")
    fig, ax = plt.subplots()
    metrics_melted = metrics_df.melt(id_vars="Model", value_vars=['AUC', 'F2'])
    sns.barplot(data=metrics_melted, x='Model', y='value', hue='variable', palette='muted')
    plt.ylim(0.7, 1.0)
    st.pyplot(fig)

    st.info("**Selection Criteria:** We chose XGBoost because it balances a high R2 for pricing accuracy with the highest F2-Score, ensuring we minimize false negatives in identifying premium listings.")
