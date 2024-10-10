import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Connect to the SQLite database
engine = create_engine('sqlite:///diagnomix.db')

def load_data():
    query = """
    SELECT * FROM prediction_result
    """
    df = pd.read_sql(query, engine)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def main():
    st.set_page_config(page_title="Diagnomix Reports", page_icon="📊", layout="wide")
    st.title("Diagnomix: Health Analysis Report")

    # Load data
    df = load_data()

    # Sidebar for date range selection
    st.sidebar.header("Filter Data")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
        min_value=df['timestamp'].min().date(),
        max_value=df['timestamp'].max().date()
    )

    start_date, end_date = date_range
    filtered_df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]

    # Display overall statistics
    st.header("Overall Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", len(filtered_df))
    col2.metric("Unique Users", filtered_df['user_id'].nunique())
    col3.metric("Date Range", f"{start_date} to {end_date}")

    # Disease-wise breakdown
    st.header("Disease-wise Breakdown")
    disease_counts = filtered_df['disease'].value_counts()
    fig_disease = px.pie(disease_counts, values=disease_counts.values, names=disease_counts.index, title="Predictions by Disease")
    st.plotly_chart(fig_disease)

    # Prediction results
    st.header("Prediction Results")
    result_counts = filtered_df['result'].value_counts()
    fig_results = px.bar(result_counts, x=result_counts.index, y=result_counts.values, title="Prediction Results")
    st.plotly_chart(fig_results)

    # Time series analysis
    st.header("Predictions Over Time")
    daily_counts = filtered_df.resample('D', on='timestamp').size().reset_index(name='count')
    fig_time = px.line(daily_counts, x='timestamp', y='count', title="Daily Prediction Counts")
    st.plotly_chart(fig_time)

    # User activity analysis
    st.header("User Activity")
    user_activity = filtered_df.groupby('user_id').size().sort_values(ascending=False)
    fig_user = px.bar(user_activity, x=user_activity.index, y=user_activity.values, title="Predictions per User")
    st.plotly_chart(fig_user)

    # Display recent predictions
    st.header("Recent Predictions")
    st.dataframe(filtered_df.sort_values('timestamp', ascending=False).head(10))

    # Export data
    if st.button("Export Data to CSV"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="diagnomix_report.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()