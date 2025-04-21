import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import hashlib
import os
from statsmodels.tsa.arima.model import ARIMA  # ARIMA Model for forecasting

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(page_title="Predictive Healthcare Analytics", layout="wide")

# ----------------- DATABASE & FILE PATH SETUP -----------------
DB_FILE = "vaccination_data.db"
USER_DB = "users.db"
DATASET_PATH = "/Users/pavansappidi/Desktop/MRP/data2.xlsx"

# Function to create database connection
def create_connection(db_path):
    return sqlite3.connect(db_path)

# Function to create user database
def setup_user_database():
    conn = create_connection(USER_DB)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE,
                        password TEXT
                      )''')
    conn.commit()
    conn.close()

# Function to create vaccination database
def setup_vaccination_database():
    conn = create_connection(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS vaccination_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        STATE TEXT,
                        CITY TEXT,
                        AGE_GROUP TEXT,
                        GENDER TEXT,
                        ETHNICITY TEXT,
                        VACCINATED BOOLEAN,
                        Year INTEGER,
                        DESCRIPTION TEXT
                      )''')
    conn.commit()
    conn.close()

# Function to check if data exists in the table
def is_data_present():
    conn = create_connection(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM vaccination_data")
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0

# Function to load dataset into the database (only if empty)
def load_data_into_db():
    if not is_data_present():
        if os.path.exists(DATASET_PATH):
            df = pd.read_excel(DATASET_PATH)  # Load from the specified path
            conn = create_connection(DB_FILE)
            df.to_sql("vaccination_data", conn, if_exists="replace", index=False)
            conn.close()
            print("✅ Data loaded into the database successfully!")
        else:
            print("❌ Error: File not found at the specified path!")

# Initialize databases
setup_user_database()
setup_vaccination_database()
load_data_into_db()

# ----------------- USER AUTHENTICATION SYSTEM -----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to check if a user exists in the database
def user_exists(username):
    conn = create_connection(USER_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

# Function to add a new user to the database
def add_user(username, password):
    conn = create_connection(USER_DB)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists

# Function to verify login credentials
def authenticate_user(username, password):
    conn = create_connection(USER_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    stored_password = cursor.fetchone()
    conn.close()
    if stored_password and stored_password[0] == hash_password(password):
        return True
    return False

# ----------------- LOGIN & SIGNUP PAGES -----------------
def login_page():
    st.title("🔑 Secure Login")
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Please try again.")

    st.write("Don't have an account?")
    if st.button("Sign Up"):
        st.session_state["signup"] = True
        st.rerun()

def signup_page():
    st.title("📝 Create a New Account")
    new_username = st.text_input("👤 Choose a Username")
    new_password = st.text_input("🔑 Choose a Password", type="password")
    confirm_password = st.text_input("🔑 Confirm Password", type="password")

    if st.button("Sign Up"):
        if new_password != confirm_password:
            st.error("❌ Passwords do not match. Try again.")
        elif user_exists(new_username):
            st.error("❌ Username already exists. Try a different one.")
        else:
            if add_user(new_username, new_password):
                st.success("✅ Account created successfully! You can now log in.")
                st.session_state["signup"] = False
                st.rerun()
            else:
                st.error("❌ Something went wrong. Try again.")

    st.write("Already have an account?")
    if st.button("Go to Login"):
        st.session_state["signup"] = False
        st.rerun()

# ----------------- AUTHENTICATION LOGIC -----------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "signup" not in st.session_state:
    st.session_state["signup"] = False

if not st.session_state["authenticated"]:
    if st.session_state["signup"]:
        signup_page()
    else:
        login_page()
    st.stop()

# ----------------- MAIN DASHBOARD -----------------
st.title("📊 Vaccination Administration and Demand Forecasting ")

# Logout Button
if st.sidebar.button("Logout"):
    st.session_state["authenticated"] = False
    st.rerun()

# ----------------- FETCH DATA FROM DATABASE -----------------
conn = create_connection(DB_FILE)
df = pd.read_sql("SELECT * FROM vaccination_data", conn)
conn.close()

st.write("### 🔍 Raw Data Preview")
st.dataframe(df.head())

# ----------------- ADD FILTERS -----------------
st.sidebar.header("🔍 Filter Data")
state = st.sidebar.selectbox("📍 Select State", df["STATE"].dropna().unique())
city = st.sidebar.selectbox("🏙 Select City", df[df["STATE"] == state]["CITY"].dropna().unique())
vaccine = st.sidebar.multiselect("💉 Select Vaccine Type", df["DESCRIPTION"].dropna().unique())

filtered_df = df[(df["STATE"] == state) & (df["CITY"] == city) & (df["DESCRIPTION"].isin(vaccine))]
st.write(f"## 📊 Data for {city}, {state} ({', '.join(vaccine)})")
st.dataframe(filtered_df)



#--------------MAP--------------------------------------------------------------------------------------------------------------------------------------------
# ----------------- MAP & SUMMARY SECTION -----------------
from geopy.geocoders import Nominatim
import geopandas as gpd
import folium
from streamlit_folium import st_folium

# ----------------- FUNCTION TO GET COORDINATES -----------------
def get_lat_lon(state, city):
    geolocator = Nominatim(user_agent="streamlit_app")
    location = geolocator.geocode(f"{city}, {state}, USA")
    if location:
        return location.latitude, location.longitude
    return None, None

# Load your corrected GeoJSON path
CITY_GEOJSON = "/Users/pavansappidi/Desktop/MRP/California_Incorporated_Cities.geojson"

try:
    # Read the GeoJSON using GeoPandas
    city_gdf = gpd.read_file(CITY_GEOJSON)

    # Filter GeoJSON for the selected city
    selected_city_boundary = city_gdf[city_gdf["CITY"].str.lower() == city.lower()]

    if not selected_city_boundary.empty:
        # Use representative point as map center
        city_center = selected_city_boundary.geometry.representative_point().iloc[0].coords[0][::-1]

        # Create Folium map centered on city
        m = folium.Map(location=city_center, zoom_start=11)

        # Add city boundary
        folium.GeoJson(
            selected_city_boundary.geometry,
            style_function=lambda x: {
                "fillOpacity": 0,
                "color": "blue",
                "weight": 3
            }
        ).add_to(m)

        st.write(f"### 🗺 City Outline for {city}")
        st_folium(m, width=800, height=500)
    else:
        st.warning(f"City '{city}' not found in GeoJSON.")
except Exception as e:
    st.error(f"Map rendering failed: {e}")




# ----------------- SHOW TOTAL VACCINATION COUNTS -----------------

# Count total vaccinated and non-vaccinated
total_vaccinated = filtered_df[filtered_df["VACCINATED"] == 1].shape[0]
total_non_vaccinated = filtered_df[filtered_df["VACCINATED"] == 0].shape[0]
total_count = total_vaccinated + total_non_vaccinated

st.write("### 🧮 Total Vaccination Status")

col1, col2, col3 = st.columns(3)
col1.metric(label="✅ Vaccinated", value=total_vaccinated)
col2.metric(label="❌ Non-Vaccinated", value=total_non_vaccinated)
col3.metric(label="📊 Total Records", value=total_count)

#--------- CoMPARISON--------------------------------------------------------------------------------------------------------------------------------------------

st.write("### 📊 Vaccination Trends: Comparison Between Vaccinated & Non-Vaccinated")

# Splitting data into Vaccinated & Non-Vaccinated groups
vaccinated_df = filtered_df[filtered_df["VACCINATED"] == 1]
non_vaccinated_df = filtered_df[filtered_df["VACCINATED"] == 0]

# Creating columns for side-by-side visualization
col1, col2 = st.columns(2)

# Ethnicity Distribution
with col1:
    st.write("### ✅ Vaccinated - Ethnicity Distribution")
    st.plotly_chart(px.pie(vaccinated_df, names="ETHNICITY", title="Vaccinated Ethnicity Distribution"))

with col2:
    st.write("### ❌ Non-Vaccinated - Ethnicity Distribution")
    st.plotly_chart(px.pie(non_vaccinated_df, names="ETHNICITY", title="Non-Vaccinated Ethnicity Distribution"))

# Gender Distribution
col3, col4 = st.columns(2)
with col3:
    st.write("### ✅ Vaccinated - Gender Distribution")
    st.plotly_chart(px.pie(vaccinated_df, names="GENDER", title="Vaccinated Gender Distribution"))

with col4:
    st.write("### ❌ Non-Vaccinated - Gender Distribution")
    st.plotly_chart(px.pie(non_vaccinated_df, names="GENDER", title="Non-Vaccinated Gender Distribution"))

# Age Group Comparison (Bar Chart)
col5, col6 = st.columns(2)
with col5:
    st.write("### ✅ Vaccinated - Age Group")
    st.plotly_chart(px.bar(vaccinated_df, x="AGE_GROUP", title="Vaccination by Age Group"))

with col6:
    st.write("### ❌ Non-Vaccinated - Age Group")
    st.plotly_chart(px.bar(non_vaccinated_df, x="AGE_GROUP", title="Non-Vaccination by Age Group"))

st.write("### 📊 Vaccination Trends (Only Vaccinated)")

# Filter only vaccinated individuals
vaccinated_df = filtered_df[filtered_df["VACCINATED"] == 1]

# ----------------- MAP ETHNICITY TO RACE (If "RACE" Column Doesn't Exist) -----------------
race_mapping = {
    "Hispanic or Latino": "Hispanic",
    "Not Hispanic or Latino": "White",
    "African American": "Black",
    "Asian": "Asian",
    "Native American": "Native American",
    "Pacific Islander": "Pacific Islander",
    "Other": "Other"
}

# If there's no "RACE" column, create one from "ETHNICITY"
if "RACE" not in vaccinated_df.columns:
    vaccinated_df["RACE"] = vaccinated_df["ETHNICITY"].map(race_mapping).fillna("Unknown")
    filtered_df["RACE"] = filtered_df["ETHNICITY"].map(race_mapping).fillna("Unknown")

# ----------------- SHOW RACE-BASED GRAPHS -----------------
st.write("### 📊 Vaccination Trend by Race")

if not vaccinated_df.empty:
    st.plotly_chart(px.bar(vaccinated_df, x="RACE", title="Vaccination by Race", color="RACE"))
else:
    st.warning("No vaccinated data available for the selected filters.")

    
st.write("### 📊 Non-Vaccination Trend by Race")

if not non_vaccinated_df.empty:
    if "RACE" not in non_vaccinated_df.columns:
        non_vaccinated_df["RACE"] = non_vaccinated_df["ETHNICITY"].map(race_mapping).fillna("Unknown")
    st.plotly_chart(px.bar(non_vaccinated_df, x="RACE", title="Non-Vaccination by Race", color="RACE"))
else:
    st.warning("No non-vaccinated data available for the selected filters.")



# ----------------- RACE-BASED BREAKDOWN TABLE -----------------
st.write("### 🧬 Vaccination vs Non-Vaccination Breakdown by Race")

# Ensure 'RACE' column exists
if "RACE" not in vaccinated_df.columns:
    vaccinated_df["RACE"] = vaccinated_df["ETHNICITY"].map(race_mapping).fillna("Unknown")
if "RACE" not in non_vaccinated_df.columns:
    non_vaccinated_df["RACE"] = non_vaccinated_df["ETHNICITY"].map(race_mapping).fillna("Unknown")

# Group by RACE
vaccinated_race_summary = vaccinated_df.groupby("RACE").size().reset_index(name="Vaccinated Count")
non_vaccinated_race_summary = non_vaccinated_df.groupby("RACE").size().reset_index(name="Non-Vaccinated Count")

# Merge summaries
race_summary_table = pd.merge(vaccinated_race_summary, non_vaccinated_race_summary, on="RACE", how="outer").fillna(0)

# Add total row
race_summary_table.loc[len(race_summary_table)] = ["Total", race_summary_table["Vaccinated Count"].sum(), race_summary_table["Non-Vaccinated Count"].sum()]

# Display table
st.dataframe(race_summary_table)



   # ----------------- SHOW SUMMARY TABLE -----------------
# Count total vaccinated and non-vaccinated
total_vaccinated = filtered_df[filtered_df["VACCINATED"] == 1].shape[0]
total_non_vaccinated = filtered_df[filtered_df["VACCINATED"] == 0].shape[0]

# Grouping data for summary
vaccinated_summary = vaccinated_df.groupby(["ETHNICITY", "GENDER", "AGE_GROUP"]).size().reset_index(name="Vaccinated Count")
non_vaccinated_summary = filtered_df[filtered_df["VACCINATED"] == 0].groupby(["ETHNICITY", "GENDER", "AGE_GROUP"]).size().reset_index(name="Non-Vaccinated Count")

# Merging vaccinated and non-vaccinated summaries
summary_table = pd.merge(vaccinated_summary, non_vaccinated_summary, on=["ETHNICITY", "GENDER", "AGE_GROUP"], how="outer").fillna(0)

# Adding total counts
summary_table.loc[len(summary_table)] = ["Total", "Total", "Total", total_vaccinated, total_non_vaccinated]

# Display Table
st.write("### 📊 Vaccination vs Non-Vaccination Breakdown")
st.dataframe(summary_table)

#----------------------------------------------------------------------------------------

# ----------------- VACCINATION FORECAST (CODE1) -----------------
st.write("### 🔮 Forecast of Vaccination Trends (Full Synthea Dataset)")

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import os

# Full Synthea data path
SYNTHETIC_DATA_PATH = "/Users/pavansappidi/Desktop/MRP/data2.xlsx"

if os.path.exists(SYNTHETIC_DATA_PATH):
    # Load only YEAR and VACCINATED columns to reduce memory
    full_df = pd.read_excel(SYNTHETIC_DATA_PATH, sheet_name="not_vaccinated_analysis (3)", usecols=["YEAR", "VACCINATED"])

    # Clean and convert
    full_df["VACCINATED"] = full_df["VACCINATED"].astype(str).str.lower().map({"true": 1, "false": 0})
    vaccinated_full = full_df[full_df["VACCINATED"] == 1]

    # Aggregate vaccinated count per year
    yearly_vax = vaccinated_full.groupby("YEAR").size().reset_index(name="vaccinated_count")

    # Fit ARIMA model
    model = ARIMA(yearly_vax["vaccinated_count"], order=(1, 1, 1))
    model_fit = model.fit()

    # Forecast next 5 years
    year_max = int(yearly_vax["YEAR"].max())
    future_years = list(range(year_max + 1, year_max + 6))
    forecast = model_fit.forecast(steps=5)

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        "YEAR": future_years,
        "vaccinated_count": forecast
    })

    # Combine with historical data
    combined_df = pd.concat([yearly_vax, forecast_df], ignore_index=True)

    # Plot the result
    fig = px.line(combined_df, x="YEAR", y="vaccinated_count",
                  title="📈 Vaccination Forecast: Historical + 5-Year Prediction",
                  markers=True)
    st.plotly_chart(fig)
    st.dataframe(combined_df)

else:
    st.warning("⚠️ Could not load full Synthea dataset for forecasting.")
#---------------


# ----------------- LOAD Census Data VACCINATION DATASET -----------------
REALTIME_DATASET_PATH = "/Users/pavansappidi/Desktop/MRP/d5f13b5b-c3c7-46ca-a8fc-ce4450a8b9cc.csv"

import numpy as np  # in case not already imported

if os.path.exists(REALTIME_DATASET_PATH):
    realtime_df = pd.read_csv(REALTIME_DATASET_PATH)

    # Compute Cenus vaccination totals
    real_fully_vaccinated = realtime_df["fully_vaccinated"].replace(np.nan, 0).sum()
    real_partially_vaccinated = realtime_df["partially_vaccinated"].replace(np.nan, 0).sum()
    real_total_vaccinated = real_fully_vaccinated + real_partially_vaccinated
else:
    st.warning("⚠️ Cenus vaccination dataset not found.")
    real_total_vaccinated = real_fully_vaccinated = real_partially_vaccinated = 0

# ----------------- SYNTHETIC VS Census Data COMPARISON -----------------
st.write("### 📊 Synthea vs Census Vaccination Comparison")

# Calculate proportions
total_population_estimate = 13802  # Replace with city/county population if known
synthea_proportion = (total_vaccinated / total_population_estimate) * 100
realtime_proportion = (real_total_vaccinated / total_population_estimate) * 100

col1, col2 = st.columns(2)
with col1:
    st.metric("✅ Synthea Vaccinated", f"{total_vaccinated}", f"{synthea_proportion:.2f}% of est. population")
with col2:
    st.metric("📡 Census Vaccinated", f"{int(real_total_vaccinated)}", f"{realtime_proportion:.2f}% of est. population")

# Optional bar chart comparison
compare_vax_df = pd.DataFrame({
    "Dataset": ["Synthea", "Census"],
    "Vaccinated": [total_vaccinated, real_total_vaccinated]
})

st.plotly_chart(
    px.bar(compare_vax_df, x="Dataset", y="Vaccinated",
           title="Vaccinated Individuals: Synthea vs Census", color="Dataset", text_auto=True)
)





#___________________________________________
# 
if real_total_vaccinated > 0:
    proportion = (total_vaccinated / real_total_vaccinated) * 100
    st.metric("📈 Synthea vs Census Proportion", f"{proportion:.2f}%", "Based on vaccination totals")
else:
    st.warning("Real-time vaccinated count is zero. Cannot calculate proportion.")







from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Ensure you have actual and predicted overlap (latest N years)
validation_years = yearly_vax["YEAR"].iloc[-5:]  # last 5 years of actuals
validation_actual = yearly_vax[yearly_vax["YEAR"].isin(validation_years)]

# Forecast values for those same years (assumes ARIMA was fit on full data)
forecast_vals = model_fit.predict(start=len(yearly_vax) - 5, end=len(yearly_vax) - 1)

# MAE & RMSE
mae = mean_absolute_error(validation_actual["vaccinated_count"], forecast_vals)
rmse = np.sqrt(mean_squared_error(validation_actual["vaccinated_count"], forecast_vals))

st.subheader("📏 Forecast Validation Metrics (Last 5 Years)")
st.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f}")




# ----------------- UNVACCINATED PROPORTION COMPARISON -----------------

# ✅ Synthea values (already available in your code)
total_population = total_vaccinated + total_non_vaccinated
unvaccinated_population_count = total_non_vaccinated

# ✅ Census values (assuming census data provides total directly or through partial)
total_population_census = real_total_vaccinated + realtime_df["unvaccinated"].replace(np.nan, 0).sum() if "unvaccinated" in realtime_df.columns else 0
unvaccinated_population_census = total_population_census - real_total_vaccinated if total_population_census > 0 else 0

# ✅ Calculate proportions
unvaccinated_proportion_synthea = (unvaccinated_population_count / total_population) * 100 if total_population > 0 else 0
unvaccinated_proportion_census = (unvaccinated_population_census / total_population_census) * 100 if total_population_census > 0 else 0

# ✅ Display side-by-side
st.write("### ❗ Unvaccinated Population Proportions (Synthea vs Census)")
col1, col2 = st.columns(2)
with col1:
    st.metric("🚫 Synthea Unvaccinated", f"{unvaccinated_proportion_synthea:.2f}%")
with col2:
    st.metric("🚫 Census Unvaccinated", f"{unvaccinated_proportion_census:.2f}%")























# ----------------- VACCINATION FORECAST (TRAIN-TEST SPLIT + VALIDATION) -----------------
st.write("### 🔮 Forecast of Vaccination Trends with Model Validation")

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.graph_objects as go

SYNTHETIC_DATA_PATH = "/Users/pavansappidi/Desktop/MRP/data2.xlsx"

if os.path.exists(SYNTHETIC_DATA_PATH):
    # Load sheet and clean
    full_df = pd.read_excel(SYNTHETIC_DATA_PATH, sheet_name="not_vaccinated_analysis (3)", usecols=["YEAR", "VACCINATED"])
    full_df["VACCINATED"] = full_df["VACCINATED"].astype(str).str.lower().map({"true": 1, "false": 0})
    vaccinated_full = full_df[full_df["VACCINATED"] == 1]

    # Group yearly
    yearly_vax = vaccinated_full.groupby("YEAR").size().reset_index(name="vaccinated_count")
    yearly_vax = yearly_vax.sort_values("YEAR")

    # Train/Test Split
    test_years = 5
    train_data = yearly_vax[:-test_years]
    test_data = yearly_vax[-test_years:]

    # Fit model on training data
    model = ARIMA(train_data["vaccinated_count"], order=(1, 1, 1))
    model_fit = model.fit()

    # Forecast for test years
    forecast = model_fit.forecast(steps=test_years)

    # Merge predicted with actuals
    test_years_list = test_data["YEAR"].tolist()
    forecast_df = pd.DataFrame({
        "YEAR": test_years_list,
        "Actual": test_data["vaccinated_count"].values,
        "Forecast": forecast
    })

    # MAE and RMSE
    mae = mean_absolute_error(forecast_df["Actual"], forecast_df["Forecast"])
    rmse = np.sqrt(mean_squared_error(forecast_df["Actual"], forecast_df["Forecast"]))

    # Show forecast vs actuals
    st.write("### 📈 Forecast vs Actual (Validation Set)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df["YEAR"], y=forecast_df["Actual"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=forecast_df["YEAR"], y=forecast_df["Forecast"], mode="lines+markers", name="Forecast"))
    fig.update_layout(title="Actual vs Forecasted Vaccination Counts (Last 5 Years)", xaxis_title="Year", yaxis_title="Vaccinated Count")
    st.plotly_chart(fig)
    st.dataframe(forecast_df)

    # Show metrics
    st.subheader("📏 Forecast Validation Metrics (Based on Test Set)")
    col1, col2 = st.columns(2)
    col1.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
    col2.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f}")

else:
    st.warning("⚠️ Could not load full Synthea dataset for forecasting.")












# ----------------- GROUPED FORECASTING (AGE, GENDER, RACE) -----------------
st.subheader("🔍 Forecasting by Group (Age, Gender, Race)")

# Create RACE column from ETHNICITY if available
if "ETHNICITY" in vaccinated_full.columns:
    race_mapping = {
        "Hispanic or Latino": "Hispanic",
        "Not Hispanic or Latino": "White",
        "African American": "Black",
        "Asian": "Asian",
        "Native American": "Native American",
        "Pacific Islander": "Pacific Islander",
        "Other": "Other"
    }
    vaccinated_full["RACE"] = vaccinated_full["ETHNICITY"].map(race_mapping).fillna("Unknown")
else:
    vaccinated_full["RACE"] = "Unknown"

# Function for grouped forecasts
def forecast_grouped_arima(df, group_col, group_name, order=(1, 1, 1), forecast_years=5):
    unique_values = df[group_col].dropna().unique()

    for value in unique_values:
        sub_df = df[df[group_col] == value]
        grouped = sub_df.groupby("YEAR").size().reset_index(name="vaccinated_count")
        grouped = grouped.sort_values("YEAR")

        if len(grouped) < 6:
            continue  # Skip if not enough years for ARIMA

        train = grouped[:-forecast_years]
        test = grouped[-forecast_years:]

        try:
            model = ARIMA(train["vaccinated_count"], order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_years)

            forecast_df = pd.DataFrame({
                "YEAR": test["YEAR"].values,
                "Actual": test["vaccinated_count"].values,
                "Forecast": forecast
            })

            st.write(f"### 📊 Forecast for {group_name}: {value}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast_df["YEAR"], y=forecast_df["Actual"], mode="lines+markers", name="Actual"))
            fig.add_trace(go.Scatter(x=forecast_df["YEAR"], y=forecast_df["Forecast"], mode="lines+markers", name="Forecast"))
            fig.update_layout(title=f"{value} - Actual vs Forecasted Vaccination", xaxis_title="Year", yaxis_title="Vaccinated Count")
            st.plotly_chart(fig)

        except Exception as e:
            st.warning(f"Could not forecast for {group_name} = {value}. Reason: {e}")

# Run group-based forecasts
forecast_grouped_arima(vaccinated_full, "AGE_GROUP", "Age Group")
forecast_grouped_arima(vaccinated_full, "GENDER", "Gender")
forecast_grouped_arima(vaccinated_full, "RACE", "Race")
