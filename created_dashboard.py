import streamlit as st
import pyodbc
import pandas as pd
import plotly.express as px
from config import VERTICA_CONFIG

st.set_page_config(page_title="DP World – Executive KPI Dashboard", layout="wide")

conn = pyodbc.connect("Driver=Vertica;Server={0};Database={1};UID={2};PWD={3}".format(
    VERTICA_CONFIG["server"], VERTICA_CONFIG["database"], VERTICA_CONFIG["uid"], VERTICA_CONFIG["pwd"]
))

def with_joined_cte(inner_query: str) -> str:
    cte = """
    WITH joined_data AS (
        SELECT v.LINE_CODE AS voyage_line_code, v.ETA_DATE, v.VOYAGE_TYPE, v.PORT_CODE, l.LINE_CODE AS line_line_code, l.LINE_NAME
        FROM DPW_DL.Voyages v
        INNER JOIN DPW_DL.DM_LINES l ON v.LINE_CODE = l.LINE_CODE
        WHERE v.VOYAGE_TYPE = 0 AND v.PORT_CODE = 'J'
    )
    """
    return cte + "\n" + inner_query

# Top 10 Shipping Lines by Vessel Count
inner_query = """
SELECT line_line_code, LINE_NAME, COUNT(*) AS vessel_count
FROM joined_data
GROUP BY line_line_code, LINE_NAME
ORDER BY vessel_count DESC
LIMIT 10
"""
query = with_joined_cte(inner_query)
df_top_lines = pd.read_sql(query, conn)
fig_top_lines = px.bar(df_top_lines, x='LINE_NAME', y='vessel_count', title='Top 10 Shipping Lines by Vessel Count', color='LINE_NAME', color_discrete_sequence=px.colors.qualitative.Set1)

# Vessel Count Distribution by Line
inner_query = """
SELECT line_line_code, LINE_NAME, COUNT(*) AS vessel_count
FROM joined_data
GROUP BY line_line_code, LINE_NAME
"""
query = with_joined_cte(inner_query)
df_vessel_distribution = pd.read_sql(query, conn)
fig_vessel_distribution = px.pie(df_vessel_distribution, names='LINE_NAME', values='vessel_count', title='Vessel Count Distribution by Line')

# Monthly Vessel Count for Top Lines
inner_query = """
SELECT line_line_code, LINE_NAME, DATE_TRUNC('month', CAST(ETA_DATE AS TIMESTAMP)) AS month, COUNT(*) AS vessel_count
FROM joined_data
GROUP BY line_line_code, LINE_NAME, month
ORDER BY month
LIMIT 100
"""
query = with_joined_cte(inner_query)
df_monthly_vessel_count = pd.read_sql(query, conn)
fig_monthly_vessel_count = px.line(df_monthly_vessel_count, x='month', y='vessel_count', color='LINE_NAME', title='Monthly Vessel Count for Top Lines', color_discrete_sequence=px.colors.qualitative.Pastel1)

# Average Vessels per Line
inner_query = """
SELECT AVG(vessel_count) AS avg_vessels_per_line
FROM (
    SELECT line_line_code, COUNT(*) AS vessel_count
    FROM joined_data
    GROUP BY line_line_code
) AS subquery
"""
query = with_joined_cte(inner_query)
df_avg_vessels = pd.read_sql(query, conn)
# Handle potential NULL/None values safely
avg_vessels_value = df_avg_vessels['avg_vessels_per_line'].iloc[0]
avg_vessels_per_line = int(avg_vessels_value) if avg_vessels_value is not None else 0

# Vessel Count by Voyage Type
inner_query = """
SELECT VOYAGE_TYPE, COUNT(*) AS vessel_count
FROM joined_data
GROUP BY VOYAGE_TYPE
"""
query = with_joined_cte(inner_query)
df_vessel_by_voyage_type = pd.read_sql(query, conn)
fig_vessel_by_voyage_type = px.bar(df_vessel_by_voyage_type, x='VOYAGE_TYPE', y='vessel_count', title='Vessel Count by Voyage Type', color='VOYAGE_TYPE', color_discrete_sequence=px.colors.qualitative.Plotly)

# Layout
st.header("DP World – Executive KPI Dashboard")

col1, col2 = st.columns(2)
with col1:
    st.metric("Average Vessels per Line", avg_vessels_per_line)
with col2:
    st.plotly_chart(fig_vessel_distribution, use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(fig_top_lines, use_container_width=True)
with col2:
    st.plotly_chart(fig_monthly_vessel_count, use_container_width=True)
with col3:
    st.plotly_chart(fig_vessel_by_voyage_type, use_container_width=True)

conn.close()
# ───────── END OF created_dashboard.py ─────────
