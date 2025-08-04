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
        SELECT v.LINE_CODE AS voyage_line_code, l.LINE_CODE AS line_line_code, l.LINE_NAME, v.START_WORK_DATE
        FROM DPW_DL.Voyages v
        INNER JOIN DPW_DL.DM_LINES l ON v.LINE_CODE = l.LINE_CODE
        WHERE v.PORT_CODE = 'J' AND v.VOYAGE_TYPE = 0
    )
    """
    return cte + "\n" + inner_query

# KPI 1: Top 10 Shipping Lines by Vessel Count
inner_query_1 = """
SELECT line_line_code, LINE_NAME, COUNT(voyage_line_code) AS vessel_count
FROM joined_data
GROUP BY line_line_code, LINE_NAME
ORDER BY vessel_count DESC
LIMIT 10
"""
query_1 = with_joined_cte(inner_query_1)
df_1 = pd.read_sql(query_1, conn)
fig1 = px.bar(df_1, x='LINE_NAME', y='vessel_count', title='Top 10 Shipping Lines by Vessel Count', color='LINE_NAME', color_discrete_sequence=px.colors.qualitative.Set1)

# KPI 2: Vessel Count Distribution by Shipping Line
inner_query_2 = """
SELECT line_line_code, LINE_NAME, COUNT(voyage_line_code) AS vessel_count
FROM joined_data
GROUP BY line_line_code, LINE_NAME
"""
query_2 = with_joined_cte(inner_query_2)
df_2 = pd.read_sql(query_2, conn)
fig2 = px.pie(df_2, values='vessel_count', names='LINE_NAME', title='Vessel Count Distribution by Shipping Line', color_discrete_sequence=px.colors.qualitative.Pastel1)

# KPI 3: Monthly Vessel Count for Top Shipping Lines
inner_query_3 = """
SELECT TO_DATE(CAST(EXTRACT(YEAR FROM START_WORK_DATE) AS VARCHAR) || LPAD(CAST(EXTRACT(MONTH FROM START_WORK_DATE) AS VARCHAR), 2, '0') || '01', 'YYYYMMDD') AS month, LINE_NAME, COUNT(voyage_line_code) AS vessel_count
FROM joined_data
GROUP BY month, LINE_NAME
ORDER BY month
LIMIT 100
"""
query_3 = with_joined_cte(inner_query_3)
df_3 = pd.read_sql(query_3, conn)
fig3 = px.line(df_3, x='month', y='vessel_count', color='LINE_NAME', title='Monthly Vessel Count for Top Shipping Lines', color_discrete_sequence=px.colors.qualitative.Plotly)

# KPI 4: Average Vessels per Shipping Line
inner_query_4 = """
SELECT AVG(vessel_count) AS avg_vessels
FROM (
    SELECT line_line_code, COUNT(voyage_line_code) AS vessel_count
    FROM joined_data
    GROUP BY line_line_code
) AS subquery
"""
query_4 = with_joined_cte(inner_query_4)
df_4 = pd.read_sql(query_4, conn)
avg_vessels = int(df_4['avg_vessels'].iloc[0])

# KPI 5: Top Shipping Lines by Vessel Count and TEUs
inner_query_5 = """
SELECT line_line_code, LINE_NAME, COUNT(voyage_line_code) AS vessel_count
FROM joined_data
GROUP BY line_line_code, LINE_NAME
ORDER BY vessel_count DESC
LIMIT 10
"""
query_5 = with_joined_cte(inner_query_5)
df_5 = pd.read_sql(query_5, conn)

# Layout
st.header("DP World – Executive KPI Dashboard")

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Average Vessels per Shipping Line", value=avg_vessels)
with col2:
    st.dataframe(df_5)

col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)
with col3:
    st.plotly_chart(fig3, use_container_width=True)

conn.close()
# ───────── END OF created_dashboard.py ─────────
