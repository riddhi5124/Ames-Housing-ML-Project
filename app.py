import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, precision_score, recall_score, fbeta_score, roc_auc_score

# Set page config
st.set_page_config(page_title="Ames Housing Analytics", layout="wide")

# --- DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_and_preprocess():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    target = pd.read_csv('target.csv')
    test_full = pd.merge(test, target, on='Order')
    
    def clean(df):
        temp = df.copy().drop(['Order', 'PID'], axis=1, errors='ignore')
        num_cols = temp.select_dtypes(include=[np.number]).columns
        temp[num_cols] = temp[num_cols].fillna(temp[num_cols].median())
        cat_cols = temp.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            temp[col] = temp[col].fillna('None')
            le = LabelEncoder()
            temp[col] = le.fit_transform(temp[col].astype(str))
        return temp

    return train, test_full, clean(train), clean(test_full)

train_raw, test_raw, train_c, test_c = load_and_preprocess()

# --- DYNAMIC MODEL TRAINING ---
@st.cache_resource
def train_and_get_metrics(train_df, test_df):
    X_train = train_df.drop('SalePrice', axis=1)
    y_train = train_df['SalePrice']
    X_test = test_df.drop('SalePrice', axis=1)
    y_test = test_df['SalePrice']
    
    median_val = y_train.median()
    y_train_cls = (y_train > median_val).astype(int)
    y_test_cls = (y_test > median_val).astype(int)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    reg_models = {
        "Linear Regression": LinearRegression().fit(X_train_s, y_train),
        "SVR": SVR(kernel='rbf', C=1e5).fit(X_train_s, y_train),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train, y_train),
        "XGBoost": GradientBoostingRegressor(n_estimators=50, random_state=42).fit(X_train, y_train)
    }

    clf_models = {
        "Linear Regression": LogisticRegression(max_iter=1000).fit(X_train_s, y_train_cls),
        "SVR": SVC(probability=True).fit(X_train_s, y_train_cls),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train_cls),
        "XGBoost": GradientBoostingClassifier(n_estimators=50, random_state=42).fit(X_train, y_train_cls)
    }

    results = []
    for name in reg_models.keys():
        xt = X_test_s if name in ["Linear Regression", "SVR"] else X_test
        y_pred_reg = reg_models[name].predict(xt)
        y_pred_cls = clf_models[name].predict(xt)
        y_prob_cls = clf_models[name].predict_proba(xt)[:, 1]
        
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, y_pred_reg),
            "R2": r2_score(y_test, y_pred_reg),
            "AUC": roc_auc_score(y_test_cls, y_prob_cls),
            "Precision": precision_score(y_test_cls, y_pred_cls),
            "Recall": recall_score(y_test_cls, y_pred_cls),
            "F2": fbeta_score(y_test_cls, y_pred_cls, beta=2)
        })

    return reg_models, clf_models, pd.DataFrame(results), median_val

reg_models, clf_models, metrics_df, median_price = train_and_get_metrics(train_c, test_c)

# --- NAVIGATION ---
page = st.sidebar.radio("Navigation", ["Overview", "Feature Selection", "Predictor", "Model Analytics"])

# PAGE 1: OVERVIEW
if page == "Overview":
    st.title("Ames Housing Market Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Listings", len(train_raw) + len(test_raw))
    m2.metric("Neighborhoods", train_raw['Neighborhood'].nunique())
    m3.metric("Avg Lot Size", f"{train_raw['Lot Area'].mean():,.0f} sf")
    m4.metric("Median Sale Price", f"${median_price:,.0f}")
    
    st.divider()
    
    st.subheader("Average Sale Price by Neighborhood")
    avg_price_nb = train_raw.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False).reset_index()
    fig_nb, ax_nb = plt.subplots(figsize=(12, 5))
    sns.barplot(data=avg_price_nb, x='Neighborhood', y='SalePrice', palette='viridis', ax=ax_nb)
    plt.xticks(rotation=90)
    st.pyplot(fig_nb)

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Price vs Living Area Analysis")
        fig1, ax1 = plt.subplots()
        # Legend enabled and moved to the bottom
        sns.scatterplot(data=train_raw, x='Gr Liv Area', y='SalePrice', hue='Neighborhood', alpha=0.6, ax=ax1, palette='magma')
        sns.move_legend(ax1, "upper left", bbox_to_anchor=(0, -0.2), ncol=3, title="Neighborhoods", frameon=False)
        st.pyplot(fig1)
    with col_r:
        st.subheader("Property Value Tiers")
        train_raw['Tier'] = pd.cut(train_raw['SalePrice'], bins=[0, 130000, 215000, np.inf], labels=['Low', 'Medium', 'Premium'])
        fig2, ax2 = plt.subplots()
        train_raw['Tier'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#440154', '#21918c', '#fde725'], ax=ax2)
        ax2.set_ylabel('')
        st.pyplot(fig2)

# PAGE 2: FEATURE SELECTION
elif page == "Feature Selection":
    st.title("Feature Importance & Selection")
    rf = reg_models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=train_c.drop('SalePrice', axis=1).columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=imp.head(10).values, y=imp.head(10).index, palette='magma', ax=ax)
    st.pyplot(fig)
    st.subheader("High-Value Property Data (Top 10 Rows)")
    top_cols = imp.head(5).index.tolist()
    st.dataframe(train_raw[top_cols + ['SalePrice']].sort_values(by='SalePrice', ascending=False).head(10))

# PAGE 3: PREDICTOR
elif page == "Predictor":
    st.title("Property Value Estimator")
    
    with st.form("price_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Location & Age")
            neighbor_select = st.selectbox("Neighborhood", sorted(train_raw['Neighborhood'].unique()))
            bldg_select = st.selectbox("Building Type", sorted(train_raw['Bldg Type'].unique()))
            # Year dropdowns as requested
            yr_built = st.selectbox("Year Built", sorted(train_raw['Year Built'].unique(), reverse=True))
            yr_remod = st.selectbox("Year Remodeled", sorted(train_raw['Year Remod/Add'].unique(), reverse=True))

        with col2:
            st.markdown("### Interior & Quality")
            q = st.slider("Overall Quality Score (1-10)", 1, 10, 6)
            k_qual_map = {"Excellent": "Ex", "Good": "Gd", "Typical/Average": "TA", "Fair": "Fa"}
            k_qual_select = st.selectbox("Kitchen Quality", list(k_qual_map.keys()))
            gr_area = st.number_input("Living Area (SqFt)", 400, 5000, 1500)
            bsmt_area = st.number_input("Basement Area (SqFt)", 0, 3000, 1000)

        with col3:
            st.markdown("### Facilities")
            baths = st.selectbox("Full Bathrooms", sorted(train_raw['Full Bath'].unique()), index=1)
            bedroom = st.selectbox("Total Bedrooms", sorted(train_raw['Bedroom AbvGr'].unique()), index=2)
            garage = st.selectbox("Garage Capacity (Cars)", sorted(train_raw['Garage Cars'].unique()), index=2)
            air = st.selectbox("Central Air Conditioning", ["Yes", "No"], index=0)
            heat_qual = st.selectbox("Heating Quality", ["Excellent", "Good", "Typical", "Fair", "Poor"])

        submitted = st.form_submit_button("Predict House Price")

    if submitted:
        input_vec = train_c.drop('SalePrice', axis=1).median().values.reshape(1, -1)
        input_df = pd.DataFrame(input_vec, columns=train_c.drop('SalePrice', axis=1).columns)
        
        input_df['Overall Qual'] = q
        input_df['Gr Liv Area'] = gr_area
        input_df['Total Bsmt SF'] = bsmt_area
        input_df['Year Built'] = yr_built
        input_df['Year Remod/Add'] = yr_remod
        input_df['Full Bath'] = baths
        input_df['Bedroom AbvGr'] = bedroom
        input_df['Garage Cars'] = garage
        input_df['Central Air'] = 1 if air == "Yes" else 0

        def encode_val(col, val):
            match = train_c[train_raw[col] == val][col]
            return match.iloc[0] if not match.empty else 0

        input_df['Neighborhood'] = encode_val('Neighborhood', neighbor_select)
        input_df['Bldg Type'] = encode_val('Bldg Type', bldg_select)
        input_df['Kitchen Qual'] = encode_val('Kitchen Qual', k_qual_map[k_qual_select])
        heat_map = {"Excellent": "Ex", "Good": "Gd", "Typical": "TA", "Fair": "Fa", "Poor": "Po"}
        input_df['Heating QC'] = encode_val('Heating QC', heat_map[heat_qual])

        price_pred = reg_models["XGBoost"].predict(input_df)[0]
        prob_pred = clf_models["XGBoost"].predict_proba(input_df)[0][1]
        
        st.divider()
        res_l, res_r = st.columns(2)
        with res_l:
            st.subheader(f"Estimated Market Value: ${price_pred:,.2f}")
            st.write(f"Valuation per SqFt: **${(price_pred/gr_area):,.2f}**")


# PAGE 4: MODEL ANALYTICS
elif page == "Model Analytics":
    st.title("Model Performance Metrics")
    st.dataframe(metrics_df.style.background_gradient(cmap='viridis', subset=['R2', 'AUC', 'F2']))
    
    fig, ax = plt.subplots()
    melted = metrics_df.melt(id_vars='Model', value_vars=['AUC', 'F2'])
    sns.barplot(data=melted, x='Model', y='value', hue='variable', palette='viridis', ax=ax)
    plt.ylim(0.7, 1.0)
    st.pyplot(fig)
