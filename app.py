import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="Nairobi House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ------------------------------
# Load the trained model
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load('models/xgboost_model.pkl')

model = load_model()


# ------------------------------
# Define the features (must match training)
# ------------------------------
top_locations = [
    'Lavington', 'Kitengela', 'Ruiru', 'Runda', 'Karen',
    'Kiambu Road', 'Kileleshwa', 'Ngong', 'Loresho', 'Ongata Rongai'
]

# We'll create the feature vector in the same order as during training
feature_cols = [
    'Size_SQM', 'Bedrooms_Num', 'Bathrooms_Num', 'Amenity_Count'
] + [f'Loc_{loc}' for loc in top_locations]

# ------------------------------
# Title and description
# ------------------------------
st.title("Nairobi House Price Predictor")
st.markdown("""
This app estimates the price of a house in Nairobi based on its characteristics.
The model was trained on **359 recent listings** from **BuyRentKenya**.
""")

# ------------------------------
# Input form

with st.form("input_form"):
    st.subheader("Property Details")

    col1, col2 = st.columns(2)
    with col1:
        size = st.number_input("Size (sqm)", min_value=10, max_value=2000, value=150, step=10)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
        amenity_count = st.number_input("Number of Amenities", min_value=0, max_value=20, value=3, step=1)

    location = st.selectbox("Location", options=["Other"] + top_locations)

    submitted = st.form_submit_button("Predict Price")

# ------------------------------
# When the form is submitted

if submitted:
    # Build feature vector
    features = []
    features.append(size)
    features.append(bedrooms)
    features.append(bathrooms)
    features.append(amenity_count)

    # Add location dummies (all 0, then set the selected one to 1 if it's in top_locations)
    for loc in top_locations:
        features.append(1 if location == loc else 0)

    # Convert to numpy array and reshape for prediction
    X_input = np.array(features).reshape(1, -1)

    # Predict
    first_pred = model.predict(X_input)[0]
    # inverse of log transform- had to log transform target during training for better performance,need to reverse it here to get actual price in millions.
    pred_millions = np.exp(first_pred)   

    # Convert to KSh (multiply by 1,000,000)
    pred_ksh = pred_millions * 1_000_000

    # MAE from test set (18.01M KSh) – load from saved metric
    mae_millions = joblib.load('/models/xgboost_mae.pkl')
    lower_bound = pred_ksh - mae_millions * 1_000_000
    upper_bound = pred_ksh + mae_millions * 1_000_000

    # ------------------------------
    # Display results

    st.success("### Prediction Results")
    st.metric("Estimated Price", f"KSh {pred_ksh:,.0f}")

    st.info(f"**Typical error range:** KSh {lower_bound:,.0f} – {upper_bound:,.0f}")
    st.caption("The range is based on the model's Mean Absolute Error (MAE) of 18.01M KSh. About 68% of predictions fall within this range.")

    # ------------------------------
    # Explanation of drivers
    # ------------------------------
    st.subheader("What drives this price?")
    st.markdown("""
    - **Location** is the biggest factor: being in Karen adds ~68M, while areas like Ongata Rongai subtract ~22M compared to an average location.
    - Each **bedroom** adds about **7.5M**, each **bathroom** about **6.7M**.
    - **Size** contributes about **46,000 KSh per square meter**.
    - The number of **amenities** had negligible impact in this dataset – quality over quantity!
    """)

    # Optionally show coefficients table
    with st.expander("Show detailed coefficients"):
        coef_data = {
            'Feature': ['Intercept'] + feature_cols,
            'Coefficient (M KSh)': [model.intercept_] + list(model.coef_)
        }
        coef_df = pd.DataFrame(coef_data).sort_values('Coefficient (M KSh)', ascending=False)
        st.dataframe(coef_df)

# ------------------------------
# Footer with disclaimer
# ------------------------------
st.markdown("---")
st.caption("""
**Disclaimer:** This is a prototype model built with limited data (359 listings). 
Predictions should be used as a rough guide only. Always consult a local real estate professional.
""")