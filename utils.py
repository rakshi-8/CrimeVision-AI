import pandas as pd
import numpy as np
import streamlit as st

def detect_anomalies(df):
    """
    Performs statistical anomaly detection on the crime dataset:
    1. Daily crime spikes (using rolling standard deviations / Z-score > 2.0)
    2. Police station surges (abnormal crime category clusters)
    3. Suspect repeat activity in short windows
    """
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
    anomalies = []
    
    # 1. Daily Spikes
    daily_counts = df.groupby('Date').size().reset_index(name='count')
    daily_counts = daily_counts.sort_values(by='Date')
    
    # Calculate rolling statistics
    daily_counts['rolling_mean'] = daily_counts['count'].rolling(window=14, min_periods=3).mean()
    daily_counts['rolling_std'] = daily_counts['count'].rolling(window=14, min_periods=3).std()
    
    # Fill initial NaNs
    daily_counts['rolling_mean'] = daily_counts['rolling_mean'].ffill().bfill()
    daily_counts['rolling_std'] = daily_counts['rolling_std'].ffill().bfill().replace(0, 1.0)
    
    daily_counts['z_score'] = (daily_counts['count'] - daily_counts['rolling_mean']) / daily_counts['rolling_std']
    
    spike_days = daily_counts[daily_counts['z_score'] > 2.0]
    
    for _, row in spike_days.tail(3).iterrows():
        anomalies.append({
            "type": "Spike",
            "severity": "High",
            "title": "Unusual Daily Incident Spike",
            "description": f"On {row['Date'].strftime('%d-%m-%Y')}, total crime reports surged to {row['count']} (Normal average: {row['rolling_mean']:.1f}). Z-Score: {row['z_score']:.2f}.",
            "timestamp": row['Date'].strftime('%Y-%m-%d')
        })
        
    # 2. Station-Category Surges (Clusters)
    # Check last 30 days crime counts per station & category vs historical monthly average
    max_date = df['Date'].max()
    thirty_days_ago = max_date - pd.Timedelta(days=30)
    
    recent_df = df[df['Date'] >= thirty_days_ago]
    historical_df = df[df['Date'] < thirty_days_ago]
    
    recent_counts = recent_df.groupby(['Police_Station', 'Crime_Category']).size().reset_index(name='recent_count')
    
    # Calculate monthly historical avg
    hist_months = max(1, (thirty_days_ago - df['Date'].min()).days / 30.0)
    hist_counts = historical_df.groupby(['Police_Station', 'Crime_Category']).size().reset_index(name='hist_total')
    hist_counts['hist_avg_monthly'] = hist_counts['hist_total'] / hist_months
    
    merged = pd.merge(recent_counts, hist_counts, on=['Police_Station', 'Crime_Category'], how='inner')
    # Filter where recent count is significantly higher than historical average (e.g. 2.5x and count > 3)
    merged['surge_factor'] = merged['recent_count'] / merged['hist_avg_monthly'].replace(0, 1.0)
    surges = merged[(merged['surge_factor'] > 2.5) & (merged['recent_count'] >= 4)]
    
    for _, row in surges.head(3).iterrows():
        anomalies.append({
            "type": "Cluster",
            "severity": "Medium",
            "title": f"Emerging {row['Crime_Category']} Cluster",
            "description": f"Police jurisdiction '{row['Police_Station']}' registered {row['recent_count']} cases of {row['Crime_Category']} in the last 30 days, representing a {row['surge_factor']:.1f}x surge compared to historical monthly baseline.",
            "timestamp": max_date.strftime('%Y-%m-%d')
        })
        
    # 3. Suspicious Financial Losses
    high_loss_df = df[df['Financial_Loss'] > 250000].sort_values(by='Date', ascending=False)
    for _, row in high_loss_df.head(2).iterrows():
        anomalies.append({
            "type": "Pattern",
            "severity": "High",
            "title": "Severe Economic Loss Incident",
            "description": f"FIR {row['FIR_ID']} in {row['District']} involving {row['Crime_Type']} reported a financial loss of Rs. {row['Financial_Loss']:,.2f}, triggering automatic high-impact investigation protocols.",
            "timestamp": pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
        })
        
    return anomalies

def generate_alerts(df):
    """
    Generates critical system alert feed.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
    alerts = []
    
    # 1. Rising cybercrime alert
    cyber_df = df[df["Crime_Category"] == "Cyber Crimes"]
    if len(cyber_df) > 0:
        cyber_df = cyber_df.sort_values(by="Date", ascending=True)
        half = len(cyber_df) // 2
        first_half = cyber_df.iloc[:half]
        second_half = cyber_df.iloc[half:]
        
        # compare counts
        growth = ((len(second_half) - len(first_half)) / len(first_half)) * 100 if len(first_half) > 0 else 0
        if growth > 10:
            alerts.append({
                "severity": "High",
                "category": "Cyber Crime Escalation",
                "title": "Rising Cybercrime Vectors",
                "description": f"Cyber crimes have increased by {growth:.1f}% state-wide. Main types include online bank fraud and identity theft in Bengaluru Urban."
            })
            
    # 2. Repeat offender activity
    repeat_suspects = df[df["Previous_Offenses"] >= 4].sort_values(by="Date", ascending=False)
    if len(repeat_suspects) > 0:
        sus = repeat_suspects.iloc[0]
        alerts.append({
            "severity": "High",
            "category": "Recidivism Alert",
            "title": f"Active Repeat Offender Detected",
            "description": f"Suspect {sus['Suspect_ID']} (implied in {sus['Previous_Offenses']} prior offenses) was linked to case {sus['FIR_ID']} at {sus['Police_Station']}."
        })
        
    # 3. New hotspot detection
    district_counts = df["District"].value_counts()
    if len(district_counts) > 0:
        top_dist = district_counts.index[0]
        alerts.append({
            "severity": "Medium",
            "category": "Geospatial Hotspot",
            "title": f"Crime Cluster in {top_dist}",
            "description": f"District {top_dist} accounts for {district_counts.iloc[0]:,} crimes, showing persistent crime density clusters around urban subdivisions."
        })
        
    # 4. Weekend vulnerability alert
    df_temp = df.copy()
    df_temp['Date'] = pd.to_datetime(df_temp['Date'])
    df_temp['DayOfWeek'] = df_temp['Date'].dt.day_name()
    weekend_count = len(df_temp[df_temp['DayOfWeek'].isin(['Saturday', 'Sunday'])])
    weekend_ratio = (weekend_count / len(df_temp)) * 100
    
    if weekend_ratio > 28.5: # standard baseline would be 2/7 = 28.57%
        alerts.append({
            "severity": "Medium",
            "category": "Temporal Vulnerability",
            "title": "Weekend Patrol Requirements",
            "description": f"Weekend incidents represent {weekend_ratio:.1f}% of state filings. Extra forces are recommended for Saturday evening patrols."
        })
        
    return alerts

def inject_futuristic_theme():
    """Injects custom CSS to override Streamlit default style and establish our theme."""
    st.markdown(
        """
        <style>
        /* Base page overrides */
        .stApp {
            background-color: #050811;
            color: #E2E8F0;
            font-family: 'Inter', 'Roboto', sans-serif;
        }
        
        /* Hide native Deploy button in top right */
        div[data-testid="stHeaderDeployButton"], button[data-testid="stHeaderDeployButton"], .stDeployButton, div[class*="stDeployButton"] {
            display: none !important;
        }
        
        /* Ensure all text labels and descriptions are bright and legible */
        label, .stWidgetLabel, p, .stSubheader, .stMarkdown, li {
            color: #E2E8F0 !important;
        }
        
        /* Selectbox and Multiselect text colors */
        div[data-baseweb="select"] div {
            color: #E2E8F0 !important;
        }
        
        /* Input fields value text */
        input {
            color: #E2E8F0 !important;
        }
        
        /* Main header text */
        h1, h2, h3, h4, h5, h6 {
            color: #00E5FF !important;
            font-weight: 700 !important;
            font-family: 'Outfit', sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #070B19 !important;
            border-right: 1px solid #1E293B;
            width: 280px !important;
        }
        section[data-testid="stSidebar"] .stMarkdown {
            color: #94A3B8 !important;
        }
        
        /* Sidebar Navigation items override */
        .st-emotion-cache-1cypcdb {
            background-color: #0A0F24 !important;
            border-radius: 8px !important;
            margin-bottom: 8px !important;
        }
        
        /* Glowing premium card class */
        .metric-card {
            background: linear-gradient(135deg, #090F26 0%, #050917 100%);
            border: 1px solid #1E3A8A;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0, 229, 255, 0.05);
            transition: all 0.3s ease-in-out;
        }
        
        .metric-card:hover {
            transform: translateY(-3px);
            border-color: #00E5FF;
            box-shadow: 0 8px 30px rgba(0, 229, 255, 0.15);
        }
        
        .metric-value {
            font-size: 32px;
            font-weight: 800;
            color: #00E5FF;
            margin-top: 8px;
            text-shadow: 0 0 10px rgba(0, 229, 255, 0.3);
        }
        
        .metric-title {
            font-size: 11px;
            font-weight: 600;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        
        .metric-change {
            font-size: 12px;
            color: #10B981;
            margin-top: 4px;
        }
        
        .metric-change.negative {
            color: #EF4444;
        }
        
        /* Custom alert cards */
        .alert-card {
            border-left: 4px solid #EF4444;
            background-color: #1A0D15;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 12px;
            border-top: 1px solid #3B1226;
            border-right: 1px solid #3B1226;
            border-bottom: 1px solid #3B1226;
        }
        
        .alert-card.medium {
            border-left-color: #F59E0B;
            background-color: #1A130D;
            border-top: 1px solid #3B2E12;
            border-right: 1px solid #3B2E12;
            border-bottom: 1px solid #3B2E12;
        }
        
        .alert-title {
            font-weight: 700;
            font-size: 14px;
            color: #F9FAFB;
            margin-bottom: 4px;
        }
        
        .alert-desc {
            font-size: 13px;
            color: #94A3B8;
            line-height: 1.4;
        }
        
        .alert-meta {
            font-size: 11px;
            color: #64748B;
            margin-top: 6px;
            text-transform: uppercase;
            font-weight: 500;
        }
        
        /* Custom buttons styling */
        .stButton>button {
            background: linear-gradient(90deg, #7209B7 0%, #3F37C9 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.2s !important;
        }
        
        .stButton>button:hover {
            box-shadow: 0 0 15px rgba(114, 9, 183, 0.4) !important;
            transform: scale(1.02) !important;
        }
        
        /* Override standard Streamlit tables and borders */
        .stDataFrame, div[data-testid="stTable"] {
            border: 1px solid #1E293B !important;
            background-color: #070B19 !important;
            border-radius: 8px;
        }
        
        /* Tabs design override */
        button[data-baseweb="tab"] {
            background-color: transparent !important;
            color: #94A3B8 !important;
            border: none !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            padding: 10px 20px !important;
        }
        
        button[aria-selected="true"] {
            color: #00E5FF !important;
            border-bottom: 2px solid #00E5FF !important;
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )

def inject_landing_styles():
    """CSS overrides specific to the Landing Page, including sidebar removal."""
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        
        .stApp {
            background-color: #03050B !important;
        }
        
        .landing-card {
            background: rgba(7, 11, 25, 0.85) !important;
            backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(0, 229, 255, 0.3) !important;
            border-radius: 16px !important;
            padding: 45px !important;
            max-width: 650px !important;
            box-shadow: 0 0 50px rgba(0, 229, 255, 0.2) !important;
            text-align: center !important;
            margin: auto !important;
            margin-top: 5vh !important;
        }
        
        .landing-title {
            color: #00E5FF !important;
            font-size: 42px !important;
            font-weight: 800 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
            margin-bottom: 5px !important;
            text-shadow: 0 0 20px rgba(0, 229, 255, 0.4) !important;
        }
        
        .landing-subtitle {
            color: #7209B7 !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            letter-spacing: 0.15em !important;
            margin-bottom: 20px !important;
        }
        
        .landing-desc {
            color: #E2E8F0 !important;
            font-size: 15px !important;
            line-height: 1.6 !important;
            margin-bottom: 30px !important;
        }
        
        /* Floating start button style */
        .start-btn button {
            background: linear-gradient(90deg, #00B4D8 0%, #7209B7 100%) !important;
            color: white !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            padding: 12px 36px !important;
            border-radius: 8px !important;
            border: none !important;
            box-shadow: 0 0 20px rgba(0, 229, 255, 0.3) !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
        }
        
        .start-btn button:hover {
            box-shadow: 0 0 35px rgba(114, 9, 183, 0.6) !important;
            transform: scale(1.05) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def inject_command_center_styles():
    """CSS overrides specific to the Command Center grid to optimize screen space."""
    st.markdown(
        """
        <style>
        /* Reduce block container margins and padding */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 0.5rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            max-width: 100% !important;
        }
        /* Compress metric card elements */
        .metric-card {
            padding: 10px 12px !important;
            border-radius: 8px !important;
            margin-bottom: 8px !important;
        }
        .metric-value {
            font-size: 24px !important;
            margin-top: 1px !important;
        }
        .metric-title {
            font-size: 9.5px !important;
        }
        .metric-change {
            font-size: 9.5px !important;
            margin-top: 0px !important;
        }
        /* Reduce padding inside columns */
        [data-testid="column"] {
            padding: 0px 4px !important;
        }
        hr {
            margin: 8px 0px !important;
        }
        .alert-card {
            padding: 8px 12px !important;
            margin-bottom: 6px !important;
        }
        .alert-title {
            font-size: 12px !important;
        }
        .alert-desc {
            font-size: 11px !important;
            line-height: 1.3 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

