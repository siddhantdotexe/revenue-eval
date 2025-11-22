# app.py - Generic Sales Modeling & Prediction Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from io import StringIO, BytesIO
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# Optional XGBoost
try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

st.set_page_config(page_title="ML Based Revenue Evauluator", layout="wide", page_icon="🛒")


# -------------------------
# Helper functions
# -------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def build_preprocessor(df, feature_cols):
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols)
    ], remainder="drop")

    return preprocessor, numeric_cols, cat_cols


def train_model(pipeline, X_train, y_train, X_val=None, y_val=None):
    pipeline.fit(X_train, y_train)
    train_preds = pipeline.predict(X_train)
    r2_tr = r2_score(y_train, train_preds)
    rmse_tr = rmse(y_train, train_preds)
    mae_tr = mean_absolute_error(y_train, train_preds)

    stats_dict = {"r2_train": r2_tr, "rmse_train": rmse_tr, "mae_train": mae_tr}
    if X_val is not None and y_val is not None:
        val_preds = pipeline.predict(X_val)
        stats_dict.update({
            "r2_val": r2_score(y_val, val_preds),
            "rmse_val": rmse(y_val, val_preds),
            "mae_val": mean_absolute_error(y_val, val_preds)
        })
    return pipeline, stats_dict


def download_link_fileobj(obj_bytes, filename, mime="text/csv"):
    return st.download_button(label=f"⬇ Download {filename}", data=obj_bytes, file_name=filename, mime=mime)


# -------------------------
# UI Layout
# -------------------------
st.title("🛒 ML Based Revenue Evauluator")
st.markdown(
    "Upload any sales dataset (CSV), pick target (sales), train a model, and predict. Exports pipeline & predictions.")

# Sidebar: Upload dataset or use sample
st.sidebar.header("Data / Model")
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample dataset (toy)", value=False)

if use_sample:
    df = px.data.tips()
    df = df.rename(columns={"total_bill": "Item_MRP", "tip": "Item_Outlet_Sales"})
    df["Outlet_Type"] = np.random.choice(["Supermarket Type1", "Grocery Store", "Supermarket Type2"], size=len(df))
    st.sidebar.success("Loaded sample dataset (tips -> adapted).")
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()
else:
    st.info("Upload a CSV to start or check 'Use sample dataset' in the sidebar.")
    st.stop()

st.subheader("Preview of uploaded data")
st.dataframe(df.head())

# Choose target
st.subheader("Choose target column (the column we will predict)")
target_col = st.selectbox("Select target (numeric)", options=df.columns.tolist(),
                          index=len(df.columns) - 1 if len(df.columns) > 0 else 0)

# Validate target numeric
if not pd.api.types.is_numeric_dtype(df[target_col]):
    st.warning("Target column is not numeric. The app requires numeric target for regression.")
    try:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        if df[target_col].isna().all():
            st.stop()
        else:
            st.info("Converted target column to numeric with coercion (NaN introduced for bad rows).")
    except Exception:
        st.stop()

# Features selection
st.subheader("Select feature columns (predictors)")
default_features = [c for c in df.columns if c != target_col]
feature_cols = st.multiselect("Features to use (choose one or more)", options=default_features,
                              default=default_features)

if len(feature_cols) == 0:
    st.error("Pick at least one feature column.")
    st.stop()

# Option to drop rows with missing target
if st.checkbox("Drop rows with missing target values", value=True):
    df = df.dropna(subset=[target_col])

# Train/test split ratio
st.subheader("Train / Validation settings")
test_size = st.slider("Validation set fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)

# Model selection
st.subheader("Model selection")
model_choice = st.selectbox("Choose regressor",
                            options=["XGBoost (fast, powerful)" if XGBOOST_AVAILABLE else "XGBoost (unavailable)",
                                     "RandomForest", "Ridge (linear)"])
if model_choice.startswith("XGBoost") and not XGBOOST_AVAILABLE:
    st.warning("XGBoost not available in this environment; pick RandomForest or Ridge.")

n_estimators = st.number_input("n_estimators (for tree models)", value=100, min_value=10, max_value=2000, step=10)

# Train button
if st.button("Train model"):
    with st.spinner("Building preprocessor and training model..."):
        # split
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

        # build preprocessor
        preprocessor, num_cols, cat_cols = build_preprocessor(df, feature_cols)

        # build regressor
        if "XGBoost" in model_choice and XGBOOST_AVAILABLE:
            reg = XGBRegressor(n_estimators=int(n_estimators), verbosity=0, n_jobs=-1, random_state=42)
        elif "RandomForest" in model_choice:
            reg = RandomForestRegressor(n_estimators=int(n_estimators), n_jobs=-1, random_state=42)
        else:
            reg = Ridge()

        # full pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", reg)
        ])

        # train
        pipeline, model_stats = train_model(pipeline, X_train, y_train, X_val, y_val)

        st.success("Training complete!")
        st.metric("Train R²", f"{model_stats['r2_train']:.4f}")
        st.metric("Val R²", f"{model_stats['r2_val']:.4f}" if "r2_val" in model_stats else "N/A")
        st.write(
            f"Train RMSE: {model_stats['rmse_train']:.4f} — Val RMSE: {model_stats.get('rmse_val', 'N/A'):.4f}" if "rmse_val" in model_stats else f"Train RMSE: {model_stats['rmse_train']:.4f}")

        # show residual plot for validation
        if "r2_val" in model_stats:
            preds_val = pipeline.predict(X_val)
            resid = y_val - preds_val
            fig = px.scatter(x=preds_val, y=resid, labels={"x": "Predicted", "y": "Residual"},
                             title="Residuals (val set)")
            st.plotly_chart(fig, use_container_width=True)

        # store pipeline in session state
        st.session_state["pipeline"] = pipeline
        st.session_state["feature_cols"] = feature_cols
        st.session_state["target_col"] = target_col
        st.session_state["model_stats"] = model_stats

# If a pipeline is present, show prediction UI
if "pipeline" in st.session_state:
    pipeline = st.session_state["pipeline"]
    st.sidebar.success("Model ready ✅")
    st.subheader("Single-row prediction (use current feature names)")

    # build form for single-row
    single_vals = {}
    cols = st.columns(3)
    for i, feat in enumerate(st.session_state["feature_cols"]):
        with cols[i % 3]:
            if pd.api.types.is_numeric_dtype(df[feat]):
                single_vals[feat] = st.number_input(f"{feat}", value=float(df[feat].dropna().median()))
            else:
                options = df[feat].dropna().unique().tolist()
                single_vals[feat] = st.selectbox(f"{feat}", options=options, index=0)

    if st.button("Predict single row"):
        single_df = pd.DataFrame([single_vals])
        try:
            pred = pipeline.predict(single_df)[0]
            st.success(f"Predicted {st.session_state['target_col']}: {pred:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Batch predictions
    st.subheader("Batch predictions (upload CSV with same feature columns)")
    batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch_pred")
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            missing = [c for c in st.session_state["feature_cols"] if c not in batch_df.columns]
            if missing:
                st.error(f"Uploaded file is missing required feature columns: {missing}")
            else:
                X_batch = batch_df[st.session_state["feature_cols"]]
                preds = pipeline.predict(X_batch)
                batch_df["Predicted_" + st.session_state["target_col"]] = preds
                st.dataframe(batch_df.head(30))
                csv_bytes = batch_df.to_csv(index=False).encode()
                download_link_fileobj(csv_bytes, "batch_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not process batch file: {e}")

# Analytics section (streamlined)
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data & Analytics")
if st.sidebar.button("Show Analytics Dashboard"):
    st.header("📊 Data Analytics Dashboard")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # ========================
    # SECTION 1: Statistical Summary
    # ========================
    st.subheader("📈 Statistical Summary")

    tab1, tab2 = st.tabs(["Descriptive Statistics", "Data Quality"])

    with tab1:
        st.write("**Basic Statistics for Numeric Columns**")
        st.dataframe(df[numeric_cols].describe())

        st.write("**Categorical Columns Summary**")
        cat_summary = []
        for col in cat_cols:
            cat_summary.append({
                "Column": col,
                "Unique Values": df[col].nunique(),
                "Most Common": df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A",
                "Missing": df[col].isna().sum()
            })
        if cat_summary:
            st.dataframe(pd.DataFrame(cat_summary))

    with tab2:
        st.write("**Missing Data Analysis**")
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_pct.values
        }).sort_values('Missing Count', ascending=False)

        st.dataframe(missing_df)

        if missing_df['Missing Count'].sum() > 0:
            fig = px.bar(missing_df[missing_df['Missing Count'] > 0],
                         x='Column', y='Missing %',
                         title="Missing Data Percentage by Column")
            st.plotly_chart(fig, use_container_width=True)

        st.write("**Outlier Detection (IQR Method)**")
        outlier_summary = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_summary.append({
                'Column': col,
                'Outlier Count': outliers,
                'Outlier %': round(outliers / len(df) * 100, 2)
            })

        outlier_df = pd.DataFrame(outlier_summary).sort_values('Outlier Count', ascending=False)
        st.dataframe(outlier_df)

    # ========================
    # SECTION 2: Key Visualizations
    # ========================
    st.subheader("📊 Key Visualizations")

    # 1. Correlation Heatmap
    st.write("**1. Correlation Heatmap**")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto='.2f', title="Correlation Matrix",
                        color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

    # 2. Distribution Plots
    st.write("**2. Distribution Analysis (Histograms with Box Plots)**")
    for col in numeric_cols:
        fig = px.histogram(df, x=col, nbins=50, marginal="box", title=f"Distribution: {col}")
        st.plotly_chart(fig, use_container_width=True)

    # 3. Box Plots for Categorical vs Numeric
    st.write("**3. Categorical Comparison**")
    if len(cat_cols) > 0 and len(numeric_cols) > 0:
        cat_sel = st.selectbox("Select Categorical Column", cat_cols, key="cat_comp")
        num_sel = st.selectbox("Select Numeric Column", numeric_cols, key="num_comp")

        fig = px.box(df, x=cat_sel, y=num_sel, title=f"{num_sel} by {cat_sel}")
        st.plotly_chart(fig, use_container_width=True)

    # 4. Scatter Plot with Trendline
    st.write("**4. Relationship Analysis (Scatter Plot with Trendline)**")
    if len(numeric_cols) >= 2:
        col1_sel = st.selectbox("X-axis", numeric_cols, key="scatter_x")
        col2_sel = st.selectbox("Y-axis", [c for c in numeric_cols if c != col1_sel], key="scatter_y")

        fig = px.scatter(df, x=col1_sel, y=col2_sel, trendline="ols",
                         title=f"{col2_sel} vs {col1_sel}")
        st.plotly_chart(fig, use_container_width=True)

    # 5. Categorical Value Counts
    st.write("**5. Categorical Distribution (Bar Charts)**")
    for col in cat_cols:
        counts = df[col].value_counts().nlargest(20).reset_index()
        counts.columns = [col, "count"]

        fig = px.bar(counts, x=col, y="count", title=f"Distribution: {col}")
        st.plotly_chart(fig, use_container_width=True)

    # 6. Target Analysis
    if target_col in df.columns:
        st.write(f"**6. Target Analysis: {target_col}**")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=target_col, nbins=50, title=f"{target_col} Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df, y=target_col, title=f"{target_col} Box Plot", points="outliers")
            st.plotly_chart(fig, use_container_width=True)

        # Target vs Features
        st.write(f"**Target ({target_col}) vs Features**")
        for col in numeric_cols:
            if col == target_col:
                continue
            fig = px.scatter(df, x=col, y=target_col, trendline="ols",
                             title=f"{target_col} vs {col}")
            st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.write("Built for flexible sales datasets with streamlined analytics.")