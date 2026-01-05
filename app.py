import streamlit as st
import pandas as pd
import joblib

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Predict house prices using a Random Forest Regression pipeline.")

# ---------------- Load Pipeline Model ----------------
@st.cache_resource
def load_pipeline():
    return joblib.load("rf_housing_model.pkl")  # 🔴 MUST match your pipeline file name

try:
    model = load_pipeline()
    st.success("✅ Model pipeline loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load model pipeline: {e}")
    st.stop()

# ---------------- User Inputs ----------------
st.subheader("Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sqft)", min_value=500, max_value=5000, value=1500)
    bedrooms = st.slider("Bedrooms", 1, 5, 3)
    bathrooms = st.slider("Bathrooms", 1, 4, 2)
    floors = st.selectbox("Floors", [1, 2, 3])

with col2:
    age = st.number_input("Age of House (Years)", min_value=0, max_value=50, value=10)
    distance = st.number_input(
        "Distance to City Center (km)", min_value=0.0, max_value=50.0, value=10.0
    )
    parking = st.selectbox("Parking Spaces", [0, 1, 2])

# ---------------- Prediction ----------------
if st.button("🔮 Predict House Price"):

    # 🔴 Column names MUST exactly match training data
    input_df = pd.DataFrame([{
        "Area_sqft": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "Age_of_House": age,
        "Distance_to_City_km": distance,
        "Parking": parking
    }])

    try:
        prediction = model.predict(input_df)[0]

        st.metric(
            label="Estimated House Price",
            value=f"₹ {prediction:.2f} Lakhs"
        )

        if prediction > 200:
            st.info("💡 High-value property")
        elif prediction < 80:
            st.warning("💡 Budget-friendly property")
        else:
            st.success("💡 Mid-range property")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Built with Streamlit & Random Forest Regression Pipeline")
