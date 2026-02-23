# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Nairobi Property Market Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('../data/clean_listings.csv')

df = load_data()

st.title("🏘️ Nairobi Property Market Insights")

# Sidebar filters
st.sidebar.header("Filters")
selected_locations = st.sidebar.multiselect(
    "Select Locations",
    options=df['Location'].unique(),
    default=df['Location'].value_counts().head(5).index.tolist()
)

filtered_df = df[df['Location'].isin(selected_locations)] if selected_locations else df

# 1. Median price by location
st.subheader("💰 Median Price by Location")
fig1 = px.bar(
    filtered_df.groupby('Location')['Price_Millions'].median().reset_index().sort_values('Price_Millions', ascending=False),
    x='Location', y='Price_Millions',
    title="Median Price (Millions KSh)"
)
st.plotly_chart(fig1, use_container_width=True)

# 2. Monthly price trend
st.subheader("📈 Monthly Price Trend")
df['YearMonth'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
monthly_avg = df.groupby('YearMonth')['Price_Millions'].mean().reset_index()
fig2 = px.line(monthly_avg, x='YearMonth', y='Price_Millions', title="Average Price Over Time")
st.plotly_chart(fig2, use_container_width=True)

# 3. Price per sqm comparison
st.subheader("📐 Price per SQM by Location")
fig3 = px.box(
    filtered_df,
    x='Location', y='Price_per_SQM',
    title="Price per SQM Distribution"
)
st.plotly_chart(fig3, use_container_width=True)

# 4. Amenity impact (top 10 amenities)
st.subheader("🏊 Top Amenities Impact")
# Recreate amenity dummies (simplified)
all_amenities = []
for x in df['Amenities'].dropna():
    if isinstance(x, str):
        all_amenities.extend([a.strip() for a in x.split(',')])
top_amenities = pd.Series(all_amenities).value_counts().head(10).index.tolist()

impact_data = []
for amenity in top_amenities:
    with_amen = df[df['Amenities'].astype(str).str.contains(amenity, na=False)]['Price_Millions'].median()
    without_amen = df[~df['Amenities'].astype(str).str.contains(amenity, na=False)]['Price_Millions'].median()
    impact_data.append({'Amenity': amenity, 'With': with_amen, 'Without': without_amen})

impact_df = pd.DataFrame(impact_data).melt(id_vars='Amenity', var_name='Group', value_name='Median Price')
fig4 = px.bar(impact_df, x='Amenity', y='Median Price', color='Group', barmode='group', title="Median Price With/Without Amenity")
st.plotly_chart(fig4, use_container_width=True)

st.caption("Data source: 359 listings scraped from BuyRentKenya")