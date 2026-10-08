
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load(
    "/content/drive/MyDrive/Ask_the_Soil/crop_recommendation_model.pkl"
)

label_encoder = joblib.load(
    "/content/drive/MyDrive/Ask_the_Soil/label_encoder.pkl"
)

st.set_page_config(
    page_title="Ask the Soil",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Ask the Soil")
st.subheader("What Should I Grow?")

st.write(
    "Enter your soil and environmental conditions "
    "to get the most suitable crops."
)

st.divider()

# Soil parameters

col1, col2 = st.columns(2)

with col1:
    N = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        max_value=200.0,
        value=90.0
    )

    P = st.number_input(
        "Phosphorus (P)",
        min_value=0.0,
        max_value=200.0,
        value=42.0
    )

    K = st.number_input(
        "Potassium (K)",
        min_value=0.0,
        max_value=250.0,
        value=43.0
    )

    ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5
    )

with col2:

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=25.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=200.0
    )

st.divider()

if st.button(
    "🌱 Recommend Crops",
    use_container_width=True
):

    input_data = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    probabilities = model.predict_proba(input_data)[0]

    top_indices = np.argsort(probabilities)[::-1][:3]

    crops = label_encoder.inverse_transform(top_indices)

    st.success(
        f"Best recommendation: **{crops[0].title()}**"
    )

    st.subheader("Top 3 Recommendations")

    for rank, index in enumerate(top_indices):

        crop = crops[rank]
        confidence = probabilities[index] * 100

        st.write(
            f"### {rank + 1}. {crop.title()}"
        )

        st.progress(
            float(confidence / 100)
        )

        st.write(
            f"Model confidence: **{confidence:.2f}%**"
        )
