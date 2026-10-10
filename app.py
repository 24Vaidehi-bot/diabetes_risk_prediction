
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "diabetes_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
DATASET_PATH = BASE_DIR / "diabetes_prediction.csv"

FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

CATEGORICAL_FEATURES = ["gender", "smoking_history"]

NUMERIC_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #07111f, #0b1728, #102334);
        color: #edf4ff;
    }
    [data-testid="stSidebar"] {
        background: #0a1422;
        border-right: 1px solid #20364c;
    }
    h1, h2, h3 {
        color: #f3f7ff !important;
    }
    p, label {
        color: #cbd8e8;
    }
    div[data-testid="stMetric"] {
        background: #112237;
        border: 1px solid #24445b;
        border-radius: 14px;
        padding: 16px;
    }
    div[data-testid="stMetricValue"] {
        color: #53e0c0;
    }
    .hero {
        padding: 24px;
        border-radius: 18px;
        background: linear-gradient(115deg, #102b43, #123d4b);
        border: 1px solid #28616a;
        margin-bottom: 20px;
    }
    .hero p {
        margin-bottom: 0;
        color: #c6dce9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOAD SAVED ARTIFACTS
# =========================================================
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("diabetes_model.pkl was not found.")

    if not SCALER_PATH.exists():
        raise FileNotFoundError("scaler.pkl was not found.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


@st.cache_data
def load_dataset():
    if not DATASET_PATH.exists():
        return None

    return pd.read_csv(DATASET_PATH)


# =========================================================
# MODEL PREPROCESSING
# =========================================================
def get_model_estimator(model):
    """Return the final estimator for a standard sklearn Pipeline."""
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1]
    return model


def get_expected_features(estimator):
    """Return the feature names stored by a fitted estimator, if present."""
    names = getattr(estimator, "feature_names_in_", None)

    if names is not None:
        return list(names)

    return None


def encode_patient(patient, expected_features):
    """
    Prepare the patient row using the feature names stored by the
    fitted model or scaler whenever available.

    Supports:
    - Raw eight-column inputs
    - One-hot encoded columns with names such as gender_Male
    - Numeric columns that retain their original feature names

    Numeric-only artifacts without feature names need the exact
    encoding used during training.
    """
    patient = patient.copy()

    # If the saved model is a pipeline, its own preprocessing
    # should be used. Do not apply the scaler a second time.
    if hasattr(model, "steps") and model.steps:
        return patient

    if expected_features is None:
        raise ValueError(
            "The saved model does not expose feature names. "
            "Its original training encoding must be confirmed before "
            "safe predictions can be made."
        )

    expected_features = list(expected_features)

    # Already matches the model's expected raw columns.
    if expected_features == FEATURES:
        return patient[expected_features]

    # Support common one-hot encoded training columns.
    # Only use this branch when the saved feature names explicitly
    # identify one-hot columns.
    has_one_hot_columns = any(
        name.startswith("gender_")
        or name.startswith("smoking_history_")
        for name in expected_features
    )

    if has_one_hot_columns:
        encoded = pd.get_dummies(
            patient,
            columns=CATEGORICAL_FEATURES,
            dtype=int,
        )

        # Match the exact columns and order from training.
        encoded = encoded.reindex(
            columns=expected_features,
            fill_value=0,
        )
        return encoded

    # If the model expects numeric versions of all eight features,
    # the categorical mapping must match the original training code.
    # These mappings are a common convention, not guaranteed to be
    # the mapping used to train your files.
    numeric_mapping = {
        "gender": {
            "Female": 0,
            "Male": 1,
            "Other": 2,
        },
        "smoking_history": {
            "never": 0,
            "No Info": 1,
            "current": 2,
            "former": 3,
            "ever": 4,
            "not current": 5,
        },
    }

    numeric_patient = patient.copy()

    for column, mapping in numeric_mapping.items():
        numeric_patient[column] = numeric_patient[column].map(mapping)

    if numeric_patient.isna().any().any():
        raise ValueError(
            "Categorical encoding produced missing values. "
            "Check the category mapping used when training the model."
        )

    if set(expected_features).issubset(numeric_patient.columns):
        return numeric_patient[expected_features]

    raise ValueError(
        "Could not match the patient's features to the model's "
        "expected features. Model expects: "
        + ", ".join(expected_features)
    )


def prepare_for_prediction(patient, model, scaler):
    """
    Apply the saved scaler only when the model is a standalone
    estimator and the scaler's expected columns can be identified.

    A pipeline should handle its own fitted preprocessing.
    """
    estimator = get_model_estimator(model)
    model_features = get_expected_features(estimator)

    prepared = encode_patient(patient, model_features)

    # The pipeline owns preprocessing if it is a fitted sklearn Pipeline.
    if hasattr(model, "steps") and model.steps:
        return prepared

    scaler_features = get_expected_features(scaler)

    # If the scaler was fitted with feature names, select those
    # exact columns and their exact order.
    if scaler_features is not None:
        scaler_features = list(scaler_features)

        if not set(scaler_features).issubset(prepared.columns):
            raise ValueError(
                "The scaler expects different columns. Expected: "
                + ", ".join(scaler_features)
            )

        scaled_input = prepared[scaler_features]
        scaled_values = scaler.transform(scaled_input)

        scaled_df = pd.DataFrame(
            scaled_values,
            columns=scaler_features,
            index=prepared.index,
        )

        # If the model expects exactly the scaled columns, return them.
        if model_features == scaler_features:
            return scaled_df

        # Preserve any unscaled columns if the training model used a
        # mixed scaled/unscaled feature matrix.
        result = prepared.copy()
        result.loc[:, scaler_features] = scaled_df
        return result[model_features]

    # Without scaler feature names, we cannot safely determine which
    # columns were scaled during training.
    raise ValueError(
        "scaler.pkl does not expose feature_names_in_. "
        "Confirm the exact training preprocessing before using it "
        "with this model. Applying it to guessed columns could "
        "produce incorrect predictions."
    )


def predict_patient(patient, model, scaler):
    prepared = prepare_for_prediction(patient, model, scaler)

    prediction = model.predict(prepared)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(prepared)[0]
        estimator = get_model_estimator(model)
        classes = list(getattr(estimator, "classes_", []))

        if 1 in classes:
            probability = float(probabilities[classes.index(1)])
        elif len(probabilities) == 2:
            probability = float(probabilities[1])

    return prediction, probability, prepared


# =========================================================
# APPLICATION HEADER
# =========================================================
st.title("🩺 Diabetes Risk Prediction")

st.markdown(
    """
    <div class="hero">
        <h2>Diabetes Prediction Dashboard</h2>
        <p>
        Predictions use the saved model and scaler files.
        Enter patient details to see the model's predicted class
        and, when supported, its estimated probability.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOAD FILES
# =========================================================
try:
    model, scaler = load_artifacts()
except Exception as exc:
    st.error("Could not load the saved model files.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

dataset = load_dataset()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("📂 Project Files")

    st.write("**Model:** `diabetes_model.pkl`")
    st.write("**Scaler:** `scaler.pkl`")

    if dataset is not None:
        st.metric("Dataset Records", f"{len(dataset):,}")
        st.write("**Dataset:** `diabetes_prediction.csv`")

        if "diabetes" in dataset.columns:
            counts = dataset["diabetes"].value_counts().sort_index()
            st.write("**Target distribution**")
            st.dataframe(
                counts.rename("Records"),
                use_container_width=True,
            )
    else:
        st.warning("diabetes_prediction.csv was not found.")

    st.caption(
        "The CSV provides dataset information. Predictions come "
        "from the saved model, not from a newly trained classifier."
    )

# =========================================================
# MODEL INFORMATION
# =========================================================
with st.expander("Model information"):
    estimator = get_model_estimator(model)
    st.write(f"**Model type:** `{type(estimator).__name__}`")
    st.write(f"**Scaler type:** `{type(scaler).__name__}`")

    expected = get_expected_features(estimator)

    if expected:
        st.write("**Model input columns:**")
        st.code(", ".join(expected))
    else:
        st.info(
            "The model does not expose feature names. "
            "Its training feature order and encoding must be confirmed."
        )

# =========================================================
# INPUT FORM
# =========================================================
st.subheader("👤 Patient Details")

with st.form("patient_form"):
    left, right = st.columns(2)

    with left:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"],
        )

        age = st.number_input(
            "Age (years)",
            min_value=0.08,
            max_value=100.0,
            value=35.0,
            step=1.0,
        )

        hypertension = st.selectbox(
            "Hypertension",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No",
        )

        heart_disease = st.selectbox(
            "Heart Disease",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No",
        )

    with right:
        smoking_history = st.selectbox(
            "Smoking History",
            [
                "never",
                "No Info",
                "current",
                "former",
                "ever",
                "not current",
            ],
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=100.0,
            value=25.0,
            step=0.1,
        )

        hba1c = st.number_input(
            "HbA1c Level",
            min_value=3.0,
            max_value=20.0,
            value=5.5,
            step=0.1,
        )

        glucose = st.number_input(
            "Blood Glucose Level (mg/dL)",
            min_value=50,
            max_value=500,
            value=100,
            step=1,
        )

    submitted = st.form_submit_button(
        "🔍 Predict Diabetes",
        use_container_width=True,
        type="primary",
    )

# =========================================================
# PREDICTION RESULTS
# =========================================================
if submitted:
    patient = pd.DataFrame(
        [[
            gender,
            age,
            hypertension,
            heart_disease,
            smoking_history,
            bmi,
            hba1c,
            glucose,
        ]],
        columns=FEATURES,
    )

    try:
        prediction, probability, prepared = predict_patient(
            patient, model, scaler
        )

        st.divider()
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Class", str(prediction))

        with col2:
            if probability is not None:
                st.metric(
                    "Class 1 Probability",
                    f"{probability:.1%}",
                )
            else:
                st.metric("Probability", "Not supported by this model")

        if str(prediction).strip().lower() in {"1", "true", "yes"}:
            st.warning(
                "The model predicts class 1. This is a dataset-based "
                "prediction, not a medical diagnosis."
            )
        elif str(prediction).strip().lower() in {"0", "false", "no"}:
            st.success(
                "The model predicts class 0. This does not rule out "
                "diabetes."
            )
        else:
            st.info(f"Model output: {prediction}")

        if probability is not None:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%"},
                    title={"text": "Estimated Probability of Class 1"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#38d9bd"},
                    },
                )
            )

            fig.update_layout(
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#edf4ff",
            )

            st.plotly_chart(fig, use_container_width=True)

        with st.expander("View submitted patient data"):
            st.dataframe(patient, use_container_width=True)

        with st.expander("View transformed model input"):
            st.dataframe(prepared, use_container_width=True)

        report = patient.copy()
        report["predicted_class"] = str(prediction)

        if probability is not None:
            report["class_1_probability"] = probability

        st.download_button(
            "⬇️ Download Prediction Report",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="diabetes_prediction_report.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error("The saved model could not process these inputs correctly.")
        st.code(f"{type(exc).__name__}: {exc}")

        st.markdown(
            """
            **What this means:** the saved model and scaler may expect
            different feature names, encoding, feature order, or scaling
            than this app currently supplies.
            """
        )

        st.info(
            "To correct this precisely, the training code used to create "
            "diabetes_model.pkl and scaler.pkl is needed. Do not change "
            "the category mappings or scale arbitrary columns until that "
            "preprocessing is confirmed."
        )

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "Educational project only. Model outputs are not medical diagnoses."
)
