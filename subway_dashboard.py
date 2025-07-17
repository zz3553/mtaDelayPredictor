import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Load your merged data
df = pd.read_csv("merged_subway_weather_with_humidity.csv", parse_dates=["date"])
df["delayed"] = df["incident_count"] > 0
df["month_period"] = df["date"].dt.to_period("M")

# Compute monthly delay rate
monthly_delay_rate = (
    df.groupby("month_period")["delayed"]
    .mean()
    .reset_index()
    .rename(columns={"delayed": "delay_rate"})
)
monthly_delay_rate["delay_rate"] *= 100

# Compute monthly avg weather
monthly_weather = (
    df.groupby("month_period")[
        ["temperature_2m_max", "precipitation_sum", "snowfall_sum", "windspeed_10m_max", "avg_humidity"]
    ]
    .mean()
    .reset_index()
)

monthly = pd.merge(monthly_delay_rate, monthly_weather, on="month_period")

# --- Streamlit UI ---

st.title("MTA Subway Delays vs Weather Conditions")

st.write("This dashboard explores how NYC subway delay rates relate to monthly weather conditions from 2020–2024.")

x_axis = st.selectbox(
    "Select Weather Variable to Plot Against Delay Rate:",
    [
        "temperature_2m_max",
        "precipitation_sum",
        "snowfall_sum",
        "windspeed_10m_max",
        "avg_humidity",
    ],
    format_func=lambda x: {
        "temperature_2m_max": "Avg Max Temperature (°F)",
        "precipitation_sum": "Avg Monthly Precipitation (mm)",
        "snowfall_sum": "Avg Monthly Snowfall (cm)",
        "windspeed_10m_max": "Avg Max Wind Speed (km/h)",
        "avg_humidity": "Avg Humidity (%)"
    }[x]
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=monthly, x=x_axis, y="delay_rate", ax=ax)
sns.regplot(data=monthly, x=x_axis, y="delay_rate", scatter=False, color="red", ax=ax)
ax.set_ylabel("% Days Delayed")
ax.set_xlabel(x_axis.replace("_", " ").title())
st.pyplot(fig)

st.markdown("---")
st.subheader("📅 Raw Monthly Data")
st.dataframe(monthly)
