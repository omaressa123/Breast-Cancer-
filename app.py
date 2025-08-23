import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from pathlib import Path

# ============================
# Page / Theming
# ============================
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide",
)

# Minimal custom styling
st.markdown(
    """
    <style>
      :root {
        --accent: #e75480; /* Pink for breast cancer */
        --bg: #0e1117;
        --card: #111827;
        --muted: #6b7280;
      }
      .accent { color: var(--accent) !important; }
      .card {
        background: var(--card);
        padding: 1rem 1.25rem; border-radius: 1rem;
        border: 1px solid rgba(255,255,255,0.06);
      }
      .small { font-size: 0.9rem; color: var(--muted); }
      .metric-pill {
        display:inline-block; padding: .35rem .65rem; border-radius: 999px;
        background: rgba(231,84,128,.1); color: var(--accent); font-weight: 600;
        border: 1px solid rgba(231,84,128,.25);
      }
      .stButton>button {
        border-radius: 12px; font-weight: 600; border: 1px solid rgba(255,255,255,0.15);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# Helpers
# ============================
@st.cache_resource(show_spinner=False)
def load_model_from_bytes(file_bytes: bytes):
    try:
        model = joblib.load(io.BytesIO(file_bytes))
        if isinstance(model, dict):
            possible_keys = ["model", "estimator", "clf", "regressor", "pipeline"]
            found = False
            for key in possible_keys:
                if key in model and hasattr(model[key], "predict"):
                    model = model[key]
                    found = True
                    break
            if not found:
                for k, v in model.items():
                    if hasattr(v, "predict"):
                        model = v
                        found = True
                        break
            if not found:
                keys_types = {k: type(v).__name__ for k, v in model.items()}
                st.error(
                    f"Failed to load model: The uploaded file contains a dictionary, not a model object.<br>"
                    f"Available keys in the dictionary: <code>{list(model.keys())}</code><br>"
                    f"Types of values: <code>{keys_types}</code><br>"
                    "Please upload a file containing a trained scikit-learn model, "
                    "or a dictionary with a key like 'model' or 'estimator' containing the model object with a <code>predict</code> method."
                )
                return None
        if not hasattr(model, "predict"):
            st.error(
                f"Failed to load model: The loaded object is of type <code>{type(model).__name__}</code> and does not have a <code>predict</code> method.<br>"
                "Please upload a file containing a trained scikit-learn model."
            )
            return None
        return model
    except ModuleNotFoundError as e:
        missing_module = str(e).split("'")[1]
        st.error(
            f"Failed to load model: Missing required Python module: {missing_module}.\n"
            f"Please install it in your environment (e.g., pip install {missing_module}) and reload the app."
        )
        return None
    except AttributeError as e:
        st.error(
            f"Failed to load model: {e}.\n"
            "This may be due to version mismatch between the environment where the model was saved and the current environment.\n"
            "Try to use the same library versions as used during model training."
        )
        return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def is_classifier(model) -> bool:
    try:
        from sklearn.base import is_classifier as sk_is_classifier
        return sk_is_classifier(model)
    except Exception:
        return hasattr(model, "predict_proba")

# ============================
# Sidebar – Model upload & options
# ============================
st.sidebar.header("⚙️ Settings")

model_file = st.sidebar.file_uploader(
    "Upload your breast cancer model (.pkl)", type=["pkl", "joblib"], key="model_uploader",
    help="Upload a scikit-learn model file (joblib or pickle format) trained for breast cancer prediction"
)

model = None
model_status = ""
inferred_n_features = None

if model_file is not None:
    try:
        model = load_model_from_bytes(model_file.getvalue())
        if model is not None:
            model_status = f"✅ Model loaded from **{model_file.name}**<br>Type: <code>{type(model).__name__}</code>"
            for attr in ["n_features_in_", "n_features_"]:
                if hasattr(model, attr):
                    inferred_n_features = int(getattr(model, attr))
                    break
        else:
            model_status = f"❌ Failed to load model from {model_file.name}. See error message above."
    except Exception as e:
        model_status = f"❌ Failed to load model from {model_file.name}: {e}"
else:
    model_status = "No model uploaded. Please upload a .pkl or .joblib file."

st.sidebar.markdown(f"**Model status:** {model_status}", unsafe_allow_html=True)

# Breast cancer feature names (from sklearn.datasets.load_breast_cancer)
BREAST_CANCER_FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

st.sidebar.divider()
st.sidebar.subheader("🧮 Input settings")

n_features = st.sidebar.number_input(
    "Number of features", min_value=1, max_value=len(BREAST_CANCER_FEATURES),
    value=int(inferred_n_features or len(BREAST_CANCER_FEATURES)), step=1, key="n_features_input",
    help="Number of input features for breast cancer model"
)

show_prob = st.sidebar.checkbox(
    "Show prediction probabilities", value=True, key="show_prob_checkbox"
)

# ============================
# Header
# ============================
st.markdown("""
# 🩺 <span class="accent">Breast Cancer Prediction</span>
A simple **Streamlit** interface to upload your breast cancer model and try predictions with your own inputs.
""", unsafe_allow_html=True)

# ============================
# Prediction Section
# ============================
st.markdown("### 🔮 Predict using the breast cancer model")

if model is None:
    st.warning("No model loaded yet. Please upload a .pkl or .joblib model file in the sidebar.")
else:
    if not hasattr(model, "predict"):
        st.error(f"Loaded model is of type {type(model)} and does not have a 'predict' method. Ensure the .pkl file contains a valid scikit-learn model.")
    else:
        problem_type = "Classification" if is_classifier(model) else "Regression"
        st.markdown(f"**Detected problem type:** {problem_type}")

        with st.form("predict_form"):
            st.markdown("**Input features (Breast Cancer)**")
            cols = st.columns(3)
            inputs = []
            for i, feature in enumerate(BREAST_CANCER_FEATURES[:n_features]):
                with cols[i % 3]:
                    value = st.number_input(
                        feature, value=0.0, step=0.01, format="%.4f", key=f"bc_feature_{i}"
                    )
                    inputs.append(value)

            submitted = st.form_submit_button("Predict")

            if submitted:
                try:
                    X = np.array(inputs, dtype=float).reshape(1, -1)
                    y_pred = model.predict(X)
                    # For sklearn breast cancer dataset: 0=malignant, 1=benign
                    label_map = {0: "Malignant", 1: "Benign"}
                    pred_label = label_map.get(y_pred[0], str(y_pred[0]))
                    st.success(f"Prediction: **{pred_label}** (class {y_pred[0]})")

                    if show_prob and hasattr(model, "predict_proba"):
                        try:
                            proba = model.predict_proba(X)[0]
                            class_labels = getattr(model, "classes_", [0, 1])
                            class_names = [label_map.get(c, str(c)) for c in class_labels]
                            prob_df = pd.DataFrame({"Class": class_names, "Probability": proba})
                            st.markdown("**Class probabilities:**")
                            st.dataframe(prob_df, use_container_width=True)
                            st.bar_chart(prob_df.set_index("Class"))
                        except Exception:
                            pass
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

# ============================
# Footer / Tips
# ============================
st.divider()
st.markdown("Finished")