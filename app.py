# app.py
"""
Streamlit app for Real Estate Investment Advisor (local-pickle version)

- Loads local model artifacts from models/
  - models/clf_pipeline.pkl
  - models/reg_pipeline.pkl
  - optional: models/scaler.pkl (only used if pipelines don't include scaling)
- No mlflow dependency required at runtime.
- Build inputs from the sidebar -> DataFrame -> pipeline.predict/predict_proba
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from typing import Optional

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

# -------------------------
# Paths
# -------------------------
MODELS_DIR = "models"
CLF_PICKLE = os.path.join(MODELS_DIR, "clf_pipeline.pkl")
REG_PICKLE = os.path.join(MODELS_DIR, "reg_pipeline.pkl")
SCALER_PICKLE = os.path.join(MODELS_DIR, "scaler.pkl")  # optional

# -------------------------
# Helper: load local artifact safely
# -------------------------
@st.cache_resource
def load_local_models():
    clf = None
    reg = None
    scaler = None
    # Load classifier pipeline
    try:
        if os.path.exists(CLF_PICKLE):
            clf = joblib.load(CLF_PICKLE)
    except Exception as e:
        st.sidebar.error(f"Failed to load classifier pickle: {e}")
    # Load regressor pipeline
    try:
        if os.path.exists(REG_PICKLE):
            reg = joblib.load(REG_PICKLE)
    except Exception as e:
        st.sidebar.error(f"Failed to load regressor pickle: {e}")
    # Load scaler only if present (we'll apply it only if pipeline doesn't have scaler)
    try:
        if os.path.exists(SCALER_PICKLE):
            scaler = joblib.load(SCALER_PICKLE)
    except Exception as e:
        st.sidebar.warning(f"Failed to load scaler pickle: {e}")
    return clf, reg, scaler

clf_model, reg_model, scaler_obj = load_local_models()

# -------------------------
# Minimal sidebar status
# -------------------------
st.sidebar.header("Model status")
st.sidebar.write("Classifier:", "✅" if clf_model is not None else "❌ (missing)")
st.sidebar.write("Regressor:", "✅" if reg_model is not None else "❌ (missing)")
st.sidebar.write("External scaler:", "✅" if scaler_obj is not None else "—")

# -------------------------
# Utils to inspect pipeline
# -------------------------
def pipeline_has_scaler(pipeline) -> bool:
    """Return True if pipeline's preprocess step contains a scaler-like transformer."""
    try:
        pre = getattr(pipeline, "named_steps", {}).get("preprocess", None)
        if pre is None:
            return False
        # ColumnTransformer style
        if hasattr(pre, "transformers_"):
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            for name, transformer, cols in pre.transformers_:
                if transformer is None:
                    continue
                # If transformer is a Pipeline
                if hasattr(transformer, "named_steps"):
                    for step in transformer.named_steps.values():
                        if isinstance(step, (StandardScaler, MinMaxScaler, RobustScaler)):
                            return True
                else:
                    if isinstance(transformer, (StandardScaler, MinMaxScaler, RobustScaler)):
                        return True
        return False
    except Exception:
        return False

def inspect_expected_features(pipeline):
    """Try to extract expected feature names from pipeline.preprocess if possible."""
    try:
        pre = getattr(pipeline, "named_steps", {}).get("preprocess", None)
        if pre is None:
            return None
        # If the ColumnTransformer exposes get_feature_names_out
        try:
            names = pre.get_feature_names_out()
            return list(names)
        except Exception:
            # fallback: build from transformers_
            feat_names = []
            if hasattr(pre, "transformers_"):
                for name, transformer, cols in pre.transformers_:
                    if name == "remainder":
                        continue
                    try:
                        if hasattr(transformer, "get_feature_names_out"):
                            names = transformer.get_feature_names_out(cols)
                        else:
                            # cols might be a list of column names
                            names = cols if isinstance(cols, (list, tuple)) else [cols]
                        feat_names.extend(list(names))
                    except Exception:
                        # best effort: include raw cols
                        if isinstance(cols, (list, tuple)):
                            feat_names.extend(cols)
                        else:
                            feat_names.append(cols)
            return feat_names if feat_names else None
    except Exception:
        return None

# -------------------------
# Build the input DataFrame
# -------------------------
def build_input_df(
    state: str, city: str, locality: str, property_type: str, bhk: int,
    size_sqft: float, price_lakhs: float, year_built: int, furnished: str,
    floor_no: int, total_floors: int, nearby_schools: int, nearby_hospitals: int,
    pta: str, parking: bool, security: bool, clubhouse: bool, garden: bool,
    gym: bool, playground: bool, pool: bool, facing: str, owner_type: str, availability: str
) -> pd.DataFrame:
    price_per_sqft = (price_lakhs * 100000.0) / max(1.0, size_sqft)
    age_of_property = max(0, 2024 - year_built)
    space_per_bhk = size_sqft / max(1, bhk)

    furnish_map = {'Unfurnished': 0, 'Semi-Furnished': 1, 'Fully-Furnished': 2}
    pta_map = {'Low': 1, 'Medium': 2, 'High': 3}
    availability_map = {'Under_Construction': 0, 'Ready_to_Move': 1}

    data = {
        'State': state,
        'City': city,
        'Locality': locality,
        'Property_Type': property_type,
        'BHK': int(bhk),
        'Size_in_SqFt': float(size_sqft),
        'Price_in_Lakhs': float(price_lakhs),
        'Price_per_SqFt': float(price_per_sqft),
        'Year_Built': int(year_built),
        'Furnished_Status': furnish_map.get(furnished, 0),
        'Floor_No': int(floor_no),
        'Total_Floors': int(total_floors),
        'Age_of_Property': int(age_of_property),
        'Nearby_Schools': int(nearby_schools),
        'Nearby_Hospitals': int(nearby_hospitals),
        'Public_Transport_Accessibility': pta_map.get(pta, 2),
        'Parking_Space': int(bool(parking)),
        'Security': int(bool(security)),
        'Availability_Status': availability_map.get(availability, 0),
        'Space_per_BHK': float(space_per_bhk),
        'Infra_Score': (nearby_schools * 0.4 + nearby_hospitals * 0.4 + pta_map.get(pta, 2) * 0.2),
        'Clubhouse': int(bool(clubhouse)),
        'Garden': int(bool(garden)),
        'Gym': int(bool(gym)),
        'Playground': int(bool(playground)),
        'Pool': int(bool(pool)),
        # owner & facing flags included because your training used them
        'Owner_Type_Broker': 1 if owner_type == 'Broker' else 0,
        'Owner_Type_Builder': 1 if owner_type == 'Builder' else 0,
        'Owner_Type_Owner': 1 if owner_type == 'Owner' else 0,
        'Facing_East': 1 if facing == 'East' else 0,
        'Facing_North': 1 if facing == 'North' else 0,
        'Facing_South': 1 if facing == 'South' else 0,
        'Facing_West': 1 if facing == 'West' else 0
    }
    return pd.DataFrame([data])

# -------------------------
# Sidebar: user inputs
# -------------------------
st.sidebar.title("Property Details")
state_list = ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Uttar Pradesh", "Other"]
property_type_list = ["Apartment", "Villa", "Independent House", "Other"]

state = st.sidebar.selectbox("State", state_list, index=0)
city = st.sidebar.text_input("City", value="")
locality = st.sidebar.text_input("Locality", value="")
property_type = st.sidebar.selectbox("Property Type", property_type_list, index=0)
bhk = st.sidebar.selectbox("BHK", options=[1,2,3,4,5], index=2)
size_sqft = st.sidebar.number_input("Size (SqFt)", min_value=200.0, max_value=20000.0, value=1000.0, step=50.0)
price_lakhs = st.sidebar.number_input("Price (Lakhs)", min_value=0.1, max_value=10000.0, value=50.0, step=0.1)
year_built = st.sidebar.number_input("Year Built", min_value=1900, max_value=2024, value=2015)
furnished = st.sidebar.selectbox("Furnished Status", options=['Unfurnished', 'Semi-Furnished', 'Fully-Furnished'], index=0)
floor_no = st.sidebar.number_input("Floor No", min_value=0, max_value=100, value=1)
total_floors = st.sidebar.number_input("Total Floors", min_value=1, max_value=200, value=5)
nearby_schools = st.sidebar.number_input("Nearby Schools (count)", min_value=0, max_value=50, value=2)
nearby_hospitals = st.sidebar.number_input("Nearby Hospitals (count)", min_value=0, max_value=50, value=1)
pta = st.sidebar.selectbox("Public Transport Accessibility", options=['Low','Medium','High'], index=1)
parking = st.sidebar.checkbox("Parking Space", value=True)
security = st.sidebar.checkbox("Security", value=True)
clubhouse = st.sidebar.checkbox("Clubhouse", value=False)
garden = st.sidebar.checkbox("Garden", value=False)
gym = st.sidebar.checkbox("Gym", value=False)
playground = st.sidebar.checkbox("Playground", value=False)
pool = st.sidebar.checkbox("Pool", value=False)
facing = st.sidebar.selectbox("Facing", options=['East','North','South','West'], index=1)
owner_type = st.sidebar.selectbox("Owner Type", options=['Broker','Builder','Owner'], index=0)
availability = st.sidebar.selectbox("Availability Status", options=['Ready_to_Move','Under_Construction'], index=0)

predict_button = st.sidebar.button("Predict")

# -------------------------
# Main: prediction & diagnostics
# -------------------------
st.title("Real Estate Investment Advisor")
st.write("Enter the property details in the sidebar, then click Predict.")

if predict_button:
    if clf_model is None or reg_model is None:
        st.error("Models not found in the models/ folder. Ensure clf_pipeline.pkl and reg_pipeline.pkl are present.")
    else:
        df_input = build_input_df(
            state=state, city=city, locality=locality, property_type=property_type,
            bhk=bhk, size_sqft=size_sqft, price_lakhs=price_lakhs, year_built=year_built,
            furnished=furnished, floor_no=floor_no, total_floors=total_floors,
            nearby_schools=nearby_schools, nearby_hospitals=nearby_hospitals,
            pta=pta, parking=parking, security=security, clubhouse=clubhouse,
            garden=garden, gym=gym, playground=playground, pool=pool,
            facing=facing, owner_type=owner_type, availability=availability
        )

        st.subheader("Input preview (raw)")
        st.dataframe(df_input.T, width=700)

        # Diagnostics: expected features from pipelines (concise)
        clf_expected = inspect_expected_features(clf_model)
        reg_expected = inspect_expected_features(reg_model)
        st.sidebar.subheader("Diagnostics (concise)")
        if clf_expected:
            missing = sorted(set(clf_expected) - set(df_input.columns))
            st.sidebar.write(f"Classifier expects ~{len(clf_expected)} features; missing sample: {missing[:6]}")
        else:
            st.sidebar.write("Classifier preprocess not inspectable (pipeline may accept raw features).")

        if reg_expected:
            missing = sorted(set(reg_expected) - set(df_input.columns))
            st.sidebar.write(f"Regressor expects ~{len(reg_expected)} features; missing sample: {missing[:6]}")
        else:
            st.sidebar.write("Regressor preprocess not inspectable (pipeline may accept raw features).")

        # Decide scaling application
        clf_has_scaler = pipeline_has_scaler(clf_model)
        reg_has_scaler = pipeline_has_scaler(reg_model)
        pipeline_scales = clf_has_scaler or reg_has_scaler

        if pipeline_scales:
            st.sidebar.write("Detected scaler inside pipeline: ✅ (not applying external scaler).")
        else:
            if scaler_obj is not None:
                st.sidebar.write("Pipeline has no scaler; external scaler available and will be used (aligned if possible).")
                # Attempt to align and apply scaler to numeric columns expected by scaler
                try:
                    scaler_feat_names = list(getattr(scaler_obj, "feature_names_in_", []))
                except Exception:
                    scaler_feat_names = []
                # heuristic numeric cols
                numeric_guess = [
                    'Size_in_SqFt','Price_in_Lakhs','Price_per_SqFt',
                    'Age_of_Property','Floor_No','Total_Floors',
                    'Nearby_Schools','Nearby_Hospitals','Space_per_BHK','Infra_Score'
                ]
                if scaler_feat_names:
                    cols_to_scale = [c for c in scaler_feat_names if c in df_input.columns]
                else:
                    cols_to_scale = [c for c in numeric_guess if c in df_input.columns]
                if len(cols_to_scale) > 0:
                    try:
                        scaled_vals = scaler_obj.transform(df_input[cols_to_scale].values)
                        df_input.loc[:, cols_to_scale] = scaled_vals
                        st.sidebar.success(f"Applied external scaler to: {cols_to_scale}")
                    except Exception as e:
                        st.sidebar.error(f"Failed to apply external scaler: {e}")
                else:
                    st.sidebar.warning("No matching columns found for external scaler; skipping scaling.")
            else:
                st.sidebar.write("No scaler found; pipeline will receive raw numeric features (ok if pipeline handles scaling).")

        # Safe predict wrappers
        def safe_predict_proba(model, X: pd.DataFrame) -> Optional[float]:
            try:
                if hasattr(model, "predict_proba"):
                    p = model.predict_proba(X)
                    # If binary classification, p[:,1] exists
                    if p.ndim == 2 and p.shape[1] >= 2:
                        return float(p[:,1][0])
                    # fallback if single-col probabilities
                    return float(p.ravel()[0])
                else:
                    # fallback: predict returns label
                    pred = model.predict(X)
                    return float(pred[0])
            except Exception as e:
                st.error(f"Classifier prediction error: {e}")
                return None

        def safe_predict_reg(model, X: pd.DataFrame) -> Optional[float]:
            try:
                pred = model.predict(X)
                return float(pred[0])
            except Exception as e:
                st.error(f"Regressor prediction error: {e}")
                return None

        # Run predictions
        try:
            invest_prob = safe_predict_proba(clf_model, df_input)
        except Exception as e:
            invest_prob = None
            st.error(f"Classifier failed: {e}")

        try:
            future_price = safe_predict_reg(reg_model, df_input)
        except Exception as e:
            future_price = None
            st.error(f"Regressor failed: {e}")

        # Show results
        st.markdown("## Prediction Results")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Investment Recommendation")
            if invest_prob is not None:
                st.metric("Good Investment Probability", f"{invest_prob:.2f}")
                st.write("Recommendation:", "**Good Investment**" if invest_prob > 0.5 else "**Not Ideal**")
            else:
                st.write("No classifier output available.")

        with c2:
            st.subheader("Projected Price after 5 Years")
            if future_price is not None:
                st.write(f"**{future_price:,.2f} Lakhs**")
            else:
                st.write("No regressor output available.")

        st.markdown("---")
        st.write("Computed Price per SqFt:", f"{(price_lakhs*100000/size_sqft):.2f}")
        st.write("Space per BHK:", f"{size_sqft/max(1,bhk):.2f} SqFt")

st.markdown("---")
