# ============================================================
# STREAMLIT APP — Sanitation Facility Access Predictor
# Column names exactly match the model's training data (with underscores).
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Sanitation Access Predictor",
    page_icon="🚽",
    layout="wide"
)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model_path = "best_model_pipeline.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please place the saved pipeline in the same directory.")
        st.stop()
    return joblib.load(model_path)

pipeline = load_model()

# ---------- HELPER: Build the exact feature set as per training ----------
def build_features(
    residence, electricity, sex, water_handwash_presence,
    water_treatment, hh_media, cooking_fuel, handwashing_obs,
    toilet_sharing, hh_size_category,
    drinking_water_source, water_time_category,
    housing_material,
    marital_status, region, water_location, children_under5, age_group,
    education, wealth, community_media, community_poverty, community_edu
):
    """
    Convert user inputs into a pandas DataFrame with the exact column names
    that the model expects.
    """
    # Map categories to numeric values
    hh_size_map = {"<4": 2, ">=4": 6}
    hh_size_num = hh_size_map[hh_size_category]

    water_time_map = {"Basic water": 10, "Limited water": 60}
    water_time_num = water_time_map[water_time_category]

    # Start with all columns set to 0 - using EXACT column names from the error
    features = {
        "Wealth_index_combined": wealth,  # 0-4
        "highest_educational_level_attained": education,  # 0-3
        "share_toilet_with_other": 1 if toilet_sharing == "Yes" else 0,
        "community_level_media_exposure": 1 if community_media == "High" else 0,
        "housing_material_status": 1 if housing_material == "Improved" else 0,
        "community_level_poverty": 1 if community_poverty == "High" else 0,
        "household_media_exposure": 1 if hh_media == "Yes" else 0,
        "type_of_place_of_residence": 1 if residence == "Urban" else 0,
        "community_level_education": 1 if community_edu == "High" else 0,
        "has_electricity": 1 if electricity == "Yes" else 0,
        "handwashing_place_observed": 1 if handwashing_obs == "Yes" else 0,
        "source_of_drinking_water": 1 if drinking_water_source == "Improved" else 0,
        "Number_of_households_Members": hh_size_num,
        "time_to_get_to_water_source_(minutes)": water_time_num,
        "sex_of_head_of_household": 1 if sex == "Male" else 0,
        "household_water_treatment": 1 if water_treatment == "Yes" else 0,
        "presence_of_water_at_hand_washing_place": 1 if water_handwash_presence == "Yes" else 0,
        "cooking_fuel_type": 1 if cooking_fuel == "Clean fuel" else 0,
    }

    # ---------- Location of water source (two dummy columns) ----------
    if water_location == "In own yard":
        features["location_of_source_for_water_in_own_yard/plot"] = 1
        features["location_of_source_for_water_elsewhere"] = 0
    elif water_location == "Elsewhere":
        features["location_of_source_for_water_in_own_yard/plot"] = 0
        features["location_of_source_for_water_elsewhere"] = 1
    else:  # "In own dwelling"
        features["location_of_source_for_water_in_own_yard/plot"] = 0
        features["location_of_source_for_water_elsewhere"] = 0

    # ---------- Age group (three dummy columns) ----------
    if age_group == "<35 years":
        features["age_group_35_years"] = 1
        features["age_group_35-60_years"] = 0
        features["age_group_60_years"] = 0
    elif age_group == ">60 years":
        features["age_group_35_years"] = 0
        features["age_group_35-60_years"] = 0
        features["age_group_60_years"] = 1
    else:  # "35-60 years"
        features["age_group_35_years"] = 0
        features["age_group_35-60_years"] = 1
        features["age_group_60_years"] = 0

    # ---------- Children under5 group (two dummy columns) ----------
    if children_under5 == "No child":
        features["children_under5_group_no_child"] = 1
        features["children_under5_group_1-2_children"] = 0
    elif children_under5 == "1-2 children":
        features["children_under5_group_no_child"] = 0
        features["children_under5_group_1-2_children"] = 1
    else:  # ">=3 children"
        features["children_under5_group_no_child"] = 0
        features["children_under5_group_1-2_children"] = 0

    # ---------- Region (eight dummy columns) ----------
    regions = {
        "Oromia": "region_oromia",
        "Tigray": "region_tigray",
        "South Ethiopia": "region_south_ethiopia",
        "Central Ethiopia": "region_central_ethiopia",
        "Sidama": "region_sidama",
        "South West Ethiopia": "region_south_west_ethiopia",
        "Amhara": "region_amhara",
        "Benishangul-Gumuz": "region_benishangul-gumuz",
    }
    for reg_name, col_name in regions.items():
        features[col_name] = 1 if region == reg_name else 0

    # ---------- Current marital status (only one dummy) ----------
    features["current_marital_status_married"] = 1 if marital_status == "Married" else 0

    # ---------- Create DataFrame with exact column order ----------
    column_order = [
        "Wealth_index_combined",
        "highest_educational_level_attained",
        "share_toilet_with_other",
        "community_level_media_exposure",
        "housing_material_status",
        "community_level_poverty",
        "household_media_exposure",
        "type_of_place_of_residence",
        "community_level_education",
        "has_electricity",
        "handwashing place observed",
        "location_of_source_for_water_in_own_yard/plot",
        "source_of_drinking_water",
        "Number_of_households_Members",
        "time_to_get_to_water_source_(minutes)",
        "location_of_source_for_water_elsewhere",
        "age_group_35-60_years",
        "sex_of_head_of_household",
        "children_under5_group_1-2_children",
        "children_under5_group_no_child",
        "household_water_treatment",
        "region_oromia",
        "age_group_35_years",
        "region_tigray",
        "age_group_60_years",
        "presence_of_water_at_hand_washing_place",
        "region_south_ethiopia",
        "current_marital_status_married",
        "region_central_ethiopia",
        "region_sidama",
        "region_south_west_ethiopia",
        "region_amhara",
        "cooking_fuel_type",
        "region_benishangul-gumuz",
    ]

    df = pd.DataFrame([features])[column_order]
    return df


# ---------- STREAMLIT UI ----------
st.title("🚽 Predicting household sanitation service status using interpretable machine learning: Evidence from the Ethiopia Demographic and Health Survey 2024–2025.")
st.markdown("""
This tool uses an XGBoost model to predict whether a household has **improved** or **unimproved** sanitation facilities.
""")

# ---- Sidebar ----
with st.sidebar:
    st.header("Enter Household Information")

    # Demographics
    st.subheader("Household Demographics")
    sex = st.radio("Sex of head of household", ["Female", "Male"])
    hh_size_category = st.radio("Household size", ["<4", ">=4"])
    age_group = st.selectbox("Age group of head", ["<35 years", "35-60 years", ">60 years"])
    marital_status = st.selectbox("Current marital status", ["Divorced", "Married", "Never married", "Widowed"])
    education = st.selectbox("Educational level", ["No education", "Primary", "Secondary", "Higher"])
    children_under5 = st.selectbox("Children under 5 in household", ["No child", "1-2 children", ">=3 children"])

    # Water & Sanitation
    st.subheader("Water & Sanitation")
    drinking_water_source = st.radio("Source of drinking water", ["Improved", "Unimproved"])
    water_location = st.radio("Location of water source", ["In own dwelling", "In own yard", "Elsewhere"])
    water_time_category = st.radio("Time to get to water source", ["Basic water", "Limited water"])
    water_treatment = st.radio("Household water treatment", ["No", "Yes"])
    toilet_sharing = st.radio("Share toilet with other households", ["No", "Yes"])
    water_handwash_presence = st.radio("Presence of water at handwashing place", ["No", "Yes"])
    handwashing_obs = st.radio("Handwashing place observed", ["No", "Yes"])

    # Socioeconomic
    st.subheader("Socioeconomic")
    wealth = st.selectbox("Wealth index", ["Poorest", "Poorer", "Middle", "Richer", "Richest"])
    electricity = st.radio("Has electricity", ["No", "Yes"])
    cooking_fuel = st.radio("Cooking fuel type", ["Solid fuel", "Clean fuel"])
    housing_material = st.radio("Housing material status", ["Unimproved", "Improved"])
    hh_media = st.radio("Household media exposure", ["No", "Yes"])

    # Community level
    st.subheader("Community Level")
    residence = st.radio("Type of place of residence", ["Rural", "Urban"])
    community_media = st.selectbox("Community level media exposure", ["Low", "High"])
    community_poverty = st.selectbox("Community level poverty", ["Low", "High"])
    community_edu = st.selectbox("Community level education", ["Low", "High"])
    region = st.selectbox("Region", [
        "Oromia", "Tigray", "South Ethiopia", "Central Ethiopia",
        "Sidama", "South West Ethiopia", "Amhara", "Benishangul-Gumuz", "Other"
    ])

    predict_btn = st.button("Predict Sanitation Access", type="primary")

# ---- Main panel ----
if predict_btn:
    education_map = {"No education": 0, "Primary": 1, "Secondary": 2, "Higher": 3}
    wealth_map = {"Poorest": 0, "Poorer": 1, "Middle": 2, "Richer": 3, "Richest": 4}

    X_input = build_features(
        residence=residence,
        electricity=electricity,
        sex=sex,
        water_handwash_presence=water_handwash_presence,
        water_treatment=water_treatment,
        hh_media=hh_media,
        cooking_fuel=cooking_fuel,
        handwashing_obs=handwashing_obs,
        toilet_sharing=toilet_sharing,
        hh_size_category=hh_size_category,
        drinking_water_source=drinking_water_source,
        water_time_category=water_time_category,
        housing_material=housing_material,
        marital_status=marital_status,
        region=region,
        water_location=water_location,
        children_under5=children_under5,
        age_group=age_group,
        education=education_map[education],
        wealth=wealth_map[wealth],
        community_media=community_media,
        community_poverty=community_poverty,
        community_edu=community_edu
    )

    try:
        prob_unimproved = pipeline.predict_proba(X_input)[0, 1]
        threshold = 0.5
        prediction = 1 if prob_unimproved >= threshold else 0
        class_label = "Unimproved" if prediction == 1 else "Improved"

        st.subheader("Prediction Result")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Class", class_label)
        col2.metric("Probability (Unimproved)", f"{prob_unimproved:.2%}")
        col3.metric("Threshold", f"{threshold:.2f}")

        st.markdown(f"""
        **Interpretation**:
        - The model predicts **{class_label.lower()}** sanitation facilities.
        - Probability of **unimproved** facilities: **{prob_unimproved:.1%}**.
        """)

        with st.expander("Show feature vector"):
            st.dataframe(X_input)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.info("Fill in all inputs and click 'Predict Sanitation Access'.")

st.markdown("---")
st.caption("Model: XGBoost | Data: Demographic and Health Survey, Ethiopia")
