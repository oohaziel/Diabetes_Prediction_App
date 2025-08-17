# ================================
# Diabetes Risk Prediction (Refined)
# ================================
# Key upgrades:
# - Relative CSV path (portable across machines)
# - Single preprocessing pipeline (zeros->NaN -> impute -> scale)
# - Cached preprocessor + models (fast!)
# - Batch predictions reuse fitted scaler/imputer (no leakage)
# - ANN architecture tunable from sidebar (layers, dropout, epochs)
# - Download button for batch results
# - Clean structure + comments for defense
# - Added feature importance visualizations for KNN and ANN

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)

# Keras (TensorFlow)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction Hub",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# Styling
# -----------------------------
def apply_custom_styling():
    custom_css = """
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f6; }
    .main .block-container { padding: 2rem 3rem; }
    h1, h2, h3 { color: #1c4e80; font-weight: 700; }
    .stButton>button {
        border: none; background-color: #007bff; color: white;
        padding: 0.75rem 1.5rem; border-radius: 8px; font-size: 1.1rem; font-weight: bold;
        transition: all 0.3s ease-in-out; width: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover { background-color: #0056b3; transform: scale(1.02); }
    @keyframes needle-animation {
        from { transform: translate(-50%, -100%) rotate(-90deg); }
        to   { transform: translate(-50%, -100%) rotate(var(--rotation-angle, -90deg)); }
    }
    .speedometer-needle { animation: needle-animation 2s cubic-bezier(0.68, -0.55, 0.27, 1.55) forwards; }
    """
    st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)


apply_custom_styling()


# -----------------------------
# Utilities
# -----------------------------
@st.cache_data
def load_data(rel_path: str = "diabetes.csv") -> pd.DataFrame:
    """Load the dataset. Replace 0s with NaN in columns where 0 means missing."""
    if not os.path.exists(rel_path):
        st.error(f"Could not find '{rel_path}'. Place diabetes.csv next to this script.")
        st.stop()
    df = pd.read_csv(rel_path)
    cols_zero_is_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_zero_is_missing] = df[cols_zero_is_missing].replace(0, np.nan)
    return df


def create_eda_visualizations(df):
    """Create and display EDA visualizations like histograms and correlation matrix."""
    st.subheader("Exploratory Data Analysis (EDA)")

    # Histograms for numerical features
    st.write("#### Histograms of Numerical Features")
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig_hist, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig_hist)

    # Correlation matrix
    st.write("#### Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)


@st.cache_resource
def build_preprocessor(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Fit imputer and scaler on the training set. Return fitted objects and split data."""
    # Split first to avoid leakage
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # Impute (mean) then scale (standard)
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    scaler = StandardScaler()

    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    feature_names = X.columns.tolist()
    return {
        "imputer": imputer,
        "scaler": scaler,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "feature_names": feature_names
    }


@st.cache_resource
def train_knn(X_train_scaled, y_train, n_neighbors=5, weights="uniform", metric="minkowski"):
    """Train & cache a KNN model."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
    model.fit(X_train_scaled, y_train)
    return model


def make_ann(input_dim: int, layer1=32, layer2=16, dropout=0.2, lr=1e-3):
    """Build an untrained ANN model with given architecture."""
    model = Sequential([
        Dense(layer1, activation='relu', input_shape=(input_dim,)),
        Dropout(dropout),
        Dense(layer2, activation='relu'),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model


@st.cache_resource
def train_ann(X_train_scaled, y_train, layer1=32, layer2=16, dropout=0.2, epochs=50, batch_size=16, lr=1e-3, verbose=0):
    """Train & cache an ANN model. Cached by hyperparameters."""
    model = make_ann(input_dim=X_train_scaled.shape[1], layer1=layer1, layer2=layer2, dropout=dropout, lr=lr)
    model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


def risk_speedometer(score: float):
    """Visual speedometer for risk, now with percentage."""
    score = float(np.clip(score, 0, 1))
    rotation_angle = score * 180 - 90
    risk_percentage = int(score * 100)

    if score < 0.3:
        color, level = "#28a745", "Low Risk"
    elif score < 0.7:
        color, level = "#ffc107", "Medium Risk"
    else:
        color, level = "#dc3545", "High Risk"

    html_code = f"""
    <div style="text-align: center;">
        <div style="position: relative; width: 250px; height: 125px; margin: auto; background: #eee; border-radius: 125px 125px 0 0; overflow: hidden; border: 2px solid #ddd;">
            <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;
                background: conic-gradient(#28a745 0deg 60deg, #ffc107 60deg 120deg, #dc3545 120deg 180deg);"></div>
            <div style="position: absolute; top: 100%; left: 50%; transform: translate(-50%, -100%); width: 220px; height: 110px; background: white; border-radius: 110px 110px 0 0;"></div>
            <div class="speedometer-needle" style="--rotation-angle: {rotation_angle}deg; position: absolute; top: 100%; left: 50%; transform-origin: bottom center; width: 4px; height: 110px; background: #333; border-radius: 4px 4px 0 0;"></div>
        </div>
        <div style="font-size: 28px; font-weight: bold; color: {color}; margin-top: 15px;">{level}</div>
        <div style="font-size: 22px; color: #555; margin-top: 5px;">Risk Score: {risk_percentage}%</div>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)


def recommendations(score: float):
    """Text recommendations based on risk."""
    st.write("---")
    st.header("⚕️ Recommendations Based on Your Risk")
    st.info("Disclaimer: This is not medical advice. Always consult a healthcare professional.")

    if score < 0.3:
        st.success("**Status: Low Risk.** Great! Keep up a healthy lifestyle.")
        st.markdown("- Maintain a balanced diet and regular exercise.\n- Annual check-ups.")
    elif score < 0.7:
        st.warning("**Status: Medium Risk.** You have some risk factors; take proactive steps.")
        st.markdown(
            "- Improve diet and activity.\n- Aim for 5–7% weight loss if overweight.\n- Know diabetes symptoms.")
    else:
        st.error("**Status: High Risk.** Please consult a healthcare professional soon.")
        st.markdown(
            "- Book a medical appointment (e.g., A1c test).\n- Monitor blood sugar regularly.\n- Consult a dietitian.")


def calculate_ann_feature_importance(model, feature_names):
    """Calculate feature importance for ANN using the absolute sum of first-layer weights."""
    weights = model.layers[0].get_weights()[0]
    importance = np.sum(np.abs(weights), axis=1)
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    return importance_df.sort_values(by='importance', ascending=False)


def calculate_knn_feature_importance(model, X_test, y_test, feature_names):
    """
    Calculate permutation feature importance for KNN.
    This is computationally expensive and is an approximation for KNN.
    """
    baseline_accuracy = model.score(X_test, y_test)
    importances = []
    for i in range(X_test.shape[1]):
        X_test_shuffled = X_test.copy()
        np.random.shuffle(X_test_shuffled[:, i])
        shuffled_accuracy = model.score(X_test_shuffled, y_test)
        importances.append(baseline_accuracy - shuffled_accuracy)

    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    return importance_df.sort_values(by='importance', ascending=False)


# -----------------------------
# Load & Preprocess (cached)
# -----------------------------
df = load_data("diabetes.csv")
pre = build_preprocessor(df, test_size=0.2, random_state=42)

X_train_scaled = pre["X_train_scaled"]
X_test_scaled = pre["X_test_scaled"]
y_train = pre["y_train"]
y_test = pre["y_test"]
FEATURES = pre["feature_names"]
imputer = pre["imputer"]
scaler = pre["scaler"]

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Home", "Batch Prediction", "Model Performance"])
st.sidebar.markdown("---")

# -----------------------------
# HOME (Single Prediction)
# -----------------------------
if page == "Home":
    st.title("🩺 Diabetes Risk Prediction")
    st.markdown("Enter your medical details to get a risk score. The model uses KNN and ANN (default settings).")

    # Default (fast) models cached
    # KNN defaults
    knn_model = train_knn(X_train_scaled, y_train, n_neighbors=5, weights="uniform", metric="minkowski")
    # ANN defaults
    ann_model = train_ann(X_train_scaled, y_train, layer1=32, layer2=16, dropout=0.2, epochs=50, batch_size=16, lr=1e-3,
                          verbose=0)

    method = st.radio("Input Method", ["Easy Sliders", "Manual Entry"], horizontal=True)

    c1, c2 = st.columns(2)
    if method == "Easy Sliders":
        with c1:
            pregnancies = st.slider("Pregnancies", 0, 20, 0)
            glucose = st.slider("Glucose (mg/dL)", 0, 250, 100)
            blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 140, 70)
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
        with c2:
            insulin = st.slider("Insulin (mu U/ml)", 0, 900, 80)
            bmi = st.slider("BMI", 0.0, 70.0, 25.0, 0.1)
            dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)
            age = st.slider("Age (years)", 0, 120, 30)
    else:
        with c1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 0, 1)
            glucose = st.number_input("Glucose (mg/dL)", 0, 250, 100, 1)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 140, 70, 1)
            skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20, 1)
        with c2:
            insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80, 1)
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0, 0.1)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)
            age = st.number_input("Age (years)", 0, 120, 30, 1)

    if st.button("Analyze My Risk"):
        # Prepare single row with feature order matching training
        user_row = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age
        ]], columns=FEATURES)

        # Apply same preprocessing: impute then scale
        user_imputed = imputer.transform(user_row)
        user_scaled = scaler.transform(user_imputed)

        # Predict
        knn_prob = knn_model.predict_proba(user_scaled)[:, 1][0]
        ann_prob = float(ann_model.predict(user_scaled, verbose=0).flatten()[0])
        risk = (knn_prob + ann_prob) / 2.0

        st.write("---")
        st.header("Prediction Results")
        risk_speedometer(risk)
        recommendations(risk)

# -----------------------------
# BATCH PREDICTION
# -----------------------------
elif page == "Batch Prediction":
    st.title("📄 Batch Diabetes Prediction & EDA")
    st.markdown(
        "Upload a CSV with the same columns as the training data. We’ll preprocess with the **same** imputer/scaler and predict with the default KNN + ANN models.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        raw = pd.read_csv(uploaded)
        st.subheader("1) Original Data")
        st.dataframe(raw)

        # EDA
        create_eda_visualizations(raw)

        st.subheader("3) Preprocess & Predict")
        # Replace 0s with NaN in key cols prior to imputation (like training)
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        common_zero_cols = [c for c in zero_cols if c in raw.columns]
        raw[common_zero_cols] = raw[common_zero_cols].replace(0, np.nan)

        # Align features to training features
        missing = [c for c in FEATURES if c not in raw.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        X_batch = raw[FEATURES].copy()

        # Use the SAME imputer and scaler fitted on training
        Xb_imputed = pd.DataFrame(imputer.transform(X_batch), columns=FEATURES)
        Xb_scaled = pd.DataFrame(scaler.transform(Xb_imputed), columns=FEATURES)

        st.info("Preprocessed (Imputed & Scaled) Data")
        st.dataframe(Xb_scaled)

        # Use cached default models for consistency with Home page
        knn_model = train_knn(X_train_scaled, y_train, n_neighbors=5, weights="uniform", metric="minkowski")
        ann_model = train_ann(X_train_scaled, y_train, layer1=32, layer2=16, dropout=0.2, epochs=50, batch_size=16,
                              lr=1e-3, verbose=0)

        knn_probs = knn_model.predict_proba(Xb_scaled)[:, 1]
        ann_probs = ann_model.predict(Xb_scaled, verbose=0).flatten()
        avg_probs = (knn_probs + ann_probs) / 2.0
        final_pred = (avg_probs >= 0.5).astype(int)

        results = raw.copy()
        results["KNN_Prob"] = np.round(knn_probs, 4)
        results["ANN_Prob"] = np.round(ann_probs, 4)
        results["Avg_Risk"] = np.round(avg_probs, 4)
        results["Final_Prediction"] = np.where(final_pred == 1, "Diabetic", "Not Diabetic")

        st.write("### 4) Results")
        st.dataframe(results)

        # Download button
        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", data=csv_bytes, file_name="diabetes_predictions.csv",
                           mime="text/csv")

# -----------------------------
# MODEL PERFORMANCE (Tuning)
# -----------------------------
elif page == "Model Performance":
    st.title("📊 Model Performance & Tuning")
    st.markdown("Tune hyperparameters and compare **KNN** vs **ANN** on the held-out test set.")

    st.sidebar.header("KNN Hyperparameters")
    k_neighbors = st.sidebar.slider("n_neighbors", 1, 25, 7, 2)
    knn_weights = st.sidebar.selectbox("weights", ["uniform", "distance"])
    knn_metric = st.sidebar.selectbox("metric", ["minkowski", "euclidean", "manhattan"])

    st.sidebar.header("ANN Hyperparameters")
    layer1 = st.sidebar.slider("Layer 1 units", 8, 256, 64, 8)
    layer2 = st.sidebar.slider("Layer 2 units", 4, 128, 32, 4)
    dropout = st.sidebar.slider("Dropout rate", 0.0, 0.6, 0.2, 0.05)
    epochs = st.sidebar.slider("Epochs", 10, 200, 80, 10)
    batch = st.sidebar.slider("Batch size", 8, 128, 16, 8)
    lr = st.sidebar.select_slider("Learning rate", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)

    # Train (cached by params)
    knn_model = train_knn(X_train_scaled, y_train, n_neighbors=k_neighbors, weights=knn_weights, metric=knn_metric)
    ann_model = train_ann(X_train_scaled, y_train, layer1=layer1, layer2=layer2, dropout=dropout, epochs=epochs,
                          batch_size=batch, lr=lr, verbose=0)

    # Predict
    y_pred_knn = knn_model.predict(X_test_scaled)
    y_pred_prob_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

    y_pred_prob_ann = ann_model.predict(X_test_scaled, verbose=0).flatten()
    y_pred_ann = (y_pred_prob_ann >= 0.5).astype(int)

    st.subheader("Performance Metrics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### K-Nearest Neighbors")
        st.text(f"Accuracy:  {accuracy_score(y_test, y_pred_knn):.3f}")
        st.text(f"Precision: {precision_score(y_test, y_pred_knn):.3f}")
        st.text(f"Recall:    {recall_score(y_test, y_pred_knn):.3f}")
        st.text(f"F1 Score:  {f1_score(y_test, y_pred_knn):.3f}")
    with c2:
        st.markdown("#### Neural Network")
        st.text(f"Accuracy:  {accuracy_score(y_test, y_pred_ann):.3f}")
        st.text(f"Precision: {precision_score(y_test, y_pred_ann):.3f}")
        st.text(f"Recall:    {recall_score(y_test, y_pred_ann):.3f}")
        st.text(f"F1 Score:  {f1_score(y_test, y_pred_ann):.3f}")

    st.write("---")
    st.subheader("Visualizations")
    v1, v2 = st.columns(2)
    with v1:
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("KNN Confusion Matrix")
        st.pyplot(fig)
    with v2:
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred_ann), annot=True, fmt='d', cmap='Greens', ax=ax)
        ax.set_title("ANN Confusion Matrix")
        st.pyplot(fig)

    # ROC curves
    fig, ax = plt.subplots()
    fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_prob_knn)
    fpr_ann, tpr_ann, _ = roc_curve(y_test, y_pred_prob_ann)
    ax.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {auc(fpr_knn, tpr_knn):.3f})")
    ax.plot(fpr_ann, tpr_ann, label=f"ANN (AUC = {auc(fpr_ann, tpr_ann):.3f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate");
    ax.set_ylabel("True Positive Rate");
    ax.set_title("ROC Curves")
    ax.legend()
    st.pyplot(fig)

    # Feature Importance Plots
    st.write("---")
    st.subheader("Model Interpretation: Feature Importance")

    col_knn_imp, col_ann_imp = st.columns(2)

    with col_knn_imp:
        st.markdown("#### Feature Importance for K-Nearest Neighbors")
        knn_importance = calculate_knn_feature_importance(knn_model, X_test_scaled, y_test, FEATURES)
        fig_knn_importance, ax_knn_importance = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=knn_importance, ax=ax_knn_importance, palette='viridis')
        ax_knn_importance.set_title('Permutation Feature Importance for KNN')
        st.pyplot(fig_knn_importance)
        st.markdown("""
        **Interpretation**: This chart shows the drop in model accuracy when a feature's values are randomly shuffled. A larger drop (longer bar) suggests that the feature is more important to the model's predictions.
        """)

    with col_ann_imp:
        st.markdown("#### Feature Importance for Neural Network")
        ann_importance = calculate_ann_feature_importance(ann_model, FEATURES)
        fig_ann_importance, ax_ann_importance = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=ann_importance, ax=ax_ann_importance, palette='plasma')
        ax_ann_importance.set_title('First-Layer Weight-Based Feature Importance for ANN')
        st.pyplot(fig_ann_importance)
        st.markdown("""
        **Interpretation**: For the Neural Network, feature importance is approximated by the absolute sum of the weights connecting each input feature to the first hidden layer. A larger sum indicates a greater influence on the model's output.
        """)

    st.markdown("--- ")
    st.subheader("GROUP MEMBERS")
    st.markdown("Elikem Joshua Mawuli — 11293951")
    st.markdown("Nana Kane Bruce Eshun — 11117122")
    st.markdown("Haziel Opoku Okyere — 11297675")
    st.markdown('Mariam Pokuaa Addo — 11016054')
    st.markdown("Christabel kabukie Chrappah - 11207632")#