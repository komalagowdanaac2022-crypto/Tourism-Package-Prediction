
import streamlit as st
import pandas as pd
import joblib
import os


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="🌴",
    layout="wide"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "best_random_forest_model.pkl"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# Application title
# --------------------------------------------------

st.title("🌴 Tourism Package Prediction")

st.write(
    "Predict whether a customer is likely to purchase "
    "the newly introduced Wellness Tourism Package."
)


# --------------------------------------------------
# Customer inputs
# --------------------------------------------------

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    customer_id = st.number_input(
        "Customer ID",
        min_value=1,
        value=1
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    duration_of_pitch = st.number_input(
        "Duration of Pitch",
        min_value=0,
        value=10
    )

    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    type_of_contact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )


with col2:

    number_of_person_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        value=2
    )

    number_of_followups = st.number_input(
        "Number of Followups",
        min_value=0,
        value=3
    )

    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )

    preferred_property_star = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Unmarried", "Divorced"]
    )

    number_of_trips = st.number_input(
        "Number of Trips",
        min_value=0,
        value=2
    )

    passport = st.selectbox(
        "Passport",
        [0, 1]
    )


col3, col4 = st.columns(2)

with col3:

    pitch_satisfaction_score = st.selectbox(
        "Pitch Satisfaction Score",
        [1, 2, 3, 4, 5]
    )

    own_car = st.selectbox(
        "Own Car",
        [0, 1]
    )

    number_of_children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        value=0
    )


with col4:

    designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=25000.0
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Tourism Package Purchase"):

    input_data = pd.DataFrame({
        "CustomerID": [customer_id],
        "Age": [age],
        "TypeofContact": [type_of_contact],
        "CityTier": [city_tier],
        "DurationOfPitch": [duration_of_pitch],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfPersonVisiting": [number_of_person_visiting],
        "NumberOfFollowups": [number_of_followups],
        "ProductPitched": [product_pitched],
        "PreferredPropertyStar": [preferred_property_star],
        "MaritalStatus": [marital_status],
        "NumberOfTrips": [number_of_trips],
        "Passport": [passport],
        "PitchSatisfactionScore": [pitch_satisfaction_score],
        "OwnCar": [own_car],
        "NumberOfChildrenVisiting": [number_of_children_visiting],
        "Designation": [designation],
        "MonthlyIncome": [monthly_income]
    })

    probability = model.predict_proba(input_data)[0][1]

    prediction = int(probability >= 0.45)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            f"✅ Customer is likely to purchase the package."
        )
    else:
        st.warning(
            f"❌ Customer is unlikely to purchase the package."
        )

    st.write(
        f"Purchase probability: **{probability:.2%}**"
    )
