import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------
# Page configuration

st.set_page_config(
    page_title="Nairobi House Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# ------------------------------
# Define the features (must match training)

top_locations = [
    'Lavington', 'Kitengela', 'Ruiru', 'Runda', 'Karen',
    'Kiambu Road', 'Kileleshwa', 'Ngong', 'Loresho', 'Ongata Rongai'
]

# We'll create the feature vector in the same order as during training
feature_cols = [
    'Size_SQM', 'Bedrooms_Num', 'Bathrooms_Num', 'Amenity_Count'
] + [f'Loc_{loc}' for loc in top_locations]


# Paths to saved artifacts
MODEL_PATH = 'models/xgboost_model.pkl'
MAE_PATH = 'models/xgboost_mae.pkl'

# ------------------------------
# Load model and MAE with caching

@st.cache_resource
def load_model():
    """Load the trained XGBoost model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_mae():
    """Load the saved Mean Absolute Error (in millions KSh)."""
    if not os.path.exists(MAE_PATH):
        st.error(f"MAE file not found at {MAE_PATH}. Please run model training first.")
        st.stop()
    return joblib.load(MAE_PATH)

model = load_model()
mae_millions = load_mae()   # e.g., 15.21


# ------------------------------
# Title and description

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

    location = st.selectbox("Location", options=["Other"] + top_locations,
                            help="Select the location of the property. 'Other' is for locations not in the top 10.")

    submitted = st.form_submit_button("Predict Price")

# ------------------------------
# When the form is submitted

if submitted:
    # Build feature vector in the exact order
    features = [size, bedrooms, bathrooms, amenity_count]

    # Add location dummies (all 0, then set the selected one to 1 if it's in top_locations)
    for loc in top_locations:
        features.append(1 if location == loc else 0)

    # Convert to DataFrame for model input
    X_input = pd.DataFrame([features], columns=feature_cols)

    # Predict (model predicts log(price))

    try:
        first_pred = model.predict(X_input)[0]
        # inverse of log transform- had to log transform target during training for better performance,need to reverse it here to get actual price in millions.
        pred_millions = np.exp(first_pred)   

        # Convert to KSh (multiply by 1,000,000)
        pred_ksh = pred_millions * 1_000_000
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # Calculate confidence interval using MAE
    lower_millions = pred_millions - mae_millions
    upper_millions = pred_millions + mae_millions
    lower_ksh = lower_millions * 1_000_000
    upper_ksh = upper_millions * 1_000_000

    # ------------------------------
    # Display results

    st.success("###Prediction Results: ")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estimated Price", f"KSh {pred_ksh:,.0f}")
    with col2:
        st.metric("Lower Bound", f"KSh {lower_ksh:,.0f}")
    with col3:
        st.metric("Upper Bound", f"KSh {upper_ksh:,.0f}")

    st.caption(
        f"The range is based on the model's Mean Absolute Error (MAE) of "
        f"**{mae_millions:.2f}M KSh** on the test set. About 68% of predictions fall within this interval."
    )

    # ------------------------------
    # What drives the price? (using XGBoost feature importance)

    st.subheader("What Drives This Price?")
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(8)

    st.write("**Top 8 most influential features (XGBoost):**")
    st.dataframe(imp_df, use_container_width=True)

    st.markdown("""
    - **Location** dominates: Loresho, Karen, Runda, Lavington are the strongest price drivers.
    - **Size and number of bathrooms** also matter significantly.
    - **Amenity count** had near‑zero importance – quality over quantity!
    """)

    # Optional: simple note about the data
    with st.expander("ℹ About the data & model"):
        st.markdown(f"""
        - **Training data:** 359 listings scraped from BuyRentKenya.
        - **Model:** XGBoost with log‑transformed target (to handle skewed prices).
        - **Performance:**  
          - MAE: {mae_millions:.2f}M KSh  
          - RMSE: 26.58M KSh  
          - R²: 0.729  
        - **Locations:** The top 10 most frequent areas were used as dummy variables; all others fall under "Other".
        """)

# ------------------------------
# Footer
# 
st.markdown("---")
st.caption("""
**Disclaimer:** This is a prototype built with a small dataset. Predictions are estimates only.
Always consult a local real estate professional before making financial decisions.
""")
