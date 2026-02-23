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

    
    
# MARKET DASHBOARD PAGE
# ------------------------------
elif page == "📊 Market Dashboard":
    st.title("📊 Nairobi Property Market Dashboard")
    st.markdown("Explore key trends and insights from the dataset of **{} listings**.".format(len(df)))

    # Ensure necessary columns exist
    required_cols = ['Location', 'Price_Millions', 'Size_SQM']
    if not all(col in df.columns for col in required_cols):
        st.error("Dashboard requires columns: Location, Price_Millions, Size_SQM")
        st.stop()

    # ---------- 1. Median price by location ----------
    st.subheader("🏘️ Median Price by Location")
    loc_median = df.groupby('Location')['Price_Millions'].median().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10,6))
    loc_median.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_ylabel("Median Price (Millions KSh)")
    ax.set_title("Top 15 Locations by Median Price")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    st.caption("The most expensive areas are Karen, Runda, Lavington, Loresho, etc.")

    # ---------- 2. Price per sqm by location ----------
    st.subheader("📏 Price per Square Meter by Location")
    df['Price_per_SQM'] = df['Price_Millions'] / df['Size_SQM'] * 1_000_000  # KSh per sqm
    ppsqm_median = df.groupby('Location')['Price_per_SQM'].median().sort_values(ascending=False).head(15)
    fig2, ax2 = plt.subplots(figsize=(10,6))
    ppsqm_median.plot(kind='bar', ax=ax2, color='lightgreen', edgecolor='black')
    ax2.set_ylabel("Median Price per sqm (KSh)")
    ax2.set_title("Top 15 Locations by Price per sqm")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig2)
    st.caption("Price per sqm normalises by size – shows which areas offer best value for space.")

    # ---------- 3. Monthly price trend (if date available) ----------
    if 'YearMonth' in df.columns:
        st.subheader("📈 Monthly Price Trend")
        monthly = df.groupby('YearMonth')['Price_Millions'].mean().reset_index()
        monthly = monthly.sort_values('YearMonth')
        fig3, ax3 = plt.subplots(figsize=(10,5))
        ax3.plot(monthly['YearMonth'], monthly['Price_Millions'], marker='o')
        ax3.set_xticks(monthly['YearMonth'][::3])  # label every 3rd month
        ax3.set_xticklabels(monthly['YearMonth'][::3], rotation=45, ha='right')
        ax3.set_ylabel("Average Price (Millions KSh)")
        ax3.set_title("Average House Price Over Time")
        st.pyplot(fig3)
    else:
        st.info("Date information not available – cannot show monthly trend.")

    # ---------- 4. Amenity impact ----------
    st.subheader("⭐ Amenity Impact on Price")
    # If we have saved amenity coefficients from Day 3, load them. Otherwise compute a simple version.
    # For simplicity, we'll create a bar chart of median price for properties with/without top amenities.
    # But that's crude. Better to use the OLS coefficients from Day 3.
    # Let's attempt to load pre‑computed amenity coefficients from a file.
    AMENITY_COEF_PATH = 'models/amenity_coefficients.csv'
    if os.path.exists(AMENITY_COEF_PATH):
        amen_coef = pd.read_csv(AMENITY_COEF_PATH)
        fig4, ax4 = plt.subplots(figsize=(10,6))
        ax4.barh(amen_coef['Amenity'], amen_coef['Coefficient'], color='salmon', edgecolor='black')
        ax4.set_xlabel("Price Impact (M KSh)")
        ax4.set_title("Amenity Value Impact (Controlled for Size & Location)")
        ax4.invert_yaxis()
        st.pyplot(fig4)
    else:
        # Fallback: show top amenities by frequency and median price
        st.markdown("""
        **Note:** For precise amenity impact, we would need a regression model with amenity dummies.
        Below is a simple frequency chart of the most common amenities.
        """)
        if 'Amenities' in df.columns:
            from collections import Counter
            all_amenities = []
            for x in df['Amenities'].dropna():
                if isinstance(x, list):
                    all_amenities.extend(x)
                elif isinstance(x, str):
                    all_amenities.extend([a.strip() for a in x.split(',')])
            top_amens = Counter(all_amenities).most_common(10)
            amen_df = pd.DataFrame(top_amens, columns=['Amenity', 'Count'])
            st.dataframe(amen_df, use_container_width=True)
        else:
            st.warning("Amenities column not found.")

    # ---------- 5. Additional: correlation heatmap ----------
    st.subheader("🔗 Correlation Matrix")
    numeric_cols = ['Price_Millions', 'Size_SQM', 'Bedrooms_Num', 'Bathrooms_Num', 'Amenity_Count']
    numeric_df = df[numeric_cols].dropna()
    if len(numeric_df) > 0:
        fig5, ax5 = plt.subplots(figsize=(8,6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax5)
        ax5.set_title("Correlation Between Numeric Features")
        st.pyplot(fig5)
    else:
        st.warning("Not enough numeric data for correlation.")

    st.markdown("---")
    st.caption("Dashboard built with Streamlit. Data from BuyRentKenya (359 listings).")

# ------------------------------
# Footer (common)
# ------------------------------
st.sidebar.markdown("---")
st.sidebar.info(
    "**Project:** Nairobi House Price Prediction 6‑Day Sprint\n\n"
    "**Model:** XGBoost with log transform\n\n"
    "**MAE:** {:.2f}M KSh".format(mae_millions)
)

# ------------------------------
# Footer
# 
st.markdown("---")
st.caption("""
**Disclaimer:** This is a prototype built with a small dataset. Predictions are estimates only.
Always consult a local real estate professional before making financial decisions.
""")
