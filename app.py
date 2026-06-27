import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# Import our custom modules
from src.data_generator import generate_crime_dataset
from src.analytics import load_data, calculate_kpis, get_filtered_data, get_district_leaderboard
from src.models import CrimePredictorEngine
from src.network_analysis import build_crime_network, visualize_network_plotly
from src.recommender import generate_recommendations, generate_storytelling_briefs
from src.utils import detect_anomalies, generate_alerts, inject_futuristic_theme
from src.report_generator import generate_pdf_report

# Page configurations
st.set_page_config(
    page_title="CrimeVision AI - Predict. Analyze. Protect.",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Initialize data and models
DATA_PATH = "data/synthetic_crime_data.csv"

@st.cache_resource
def initialize_system():
    # Generate data if not present
    if not os.path.exists(DATA_PATH):
        generate_crime_dataset(num_records=10500, output_path=DATA_PATH)
        
    df = load_data(DATA_PATH)
    
    # Train/load ML Engine
    engine = CrimePredictorEngine(DATA_PATH)
    if not engine.load_models():
        print("Training models for the first time...")
        engine.train_risk_classifier(df)
        engine.train_volume_regressor(df)
        engine.load_models()
        
    return df, engine

try:
    df, ml_engine = initialize_system()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# Inject modern styling overrides
inject_futuristic_theme()

# Sidebar Setup
st.sidebar.markdown(
    """
    <div style='text-align: center; padding-bottom: 20px; border-bottom: 1px solid #1E293B;'>
        <h2 style='margin: 0; color: #00E5FF !important;'>CRIMEVISION AI</h2>
        <p style='margin: 5px 0 0 0; font-size: 11px; letter-spacing: 0.15em; color: #94A3B8;'>PREDICT. ANALYZE. PROTECT.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("### 🛠️ NAVIGATION")
menu = st.sidebar.radio(
    "Go To:",
    [
        "📊 Analytics Center", 
        "🗺️ Hotspot Intelligence", 
        "🔮 Predictive Intelligence", 
        "🕸️ Criminal Network Graph", 
        "🚨 Anomalies & Alert Feed", 
        "📄 Intelligence Report"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 GLOBAL FILTERS")

# Filters (State is shared across navigation tabs)
districts_list = sorted(df["District"].unique())
crime_cats_list = sorted(df["Crime_Category"].unique())
risk_levels_list = ["Low", "Medium", "High"]

# Date Range slider
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

selected_districts = st.sidebar.multiselect("Select Districts", districts_list, default=[])
selected_categories = st.sidebar.multiselect("Select Crime Categories", crime_cats_list, default=[])
selected_risks = st.sidebar.multiselect("Select Risk Levels", risk_levels_list, default=[])

date_range = st.sidebar.slider(
    "Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="DD-MM-YYYY"
)

# Apply Filters
filtered_df = get_filtered_data(
    df, 
    districts=selected_districts if selected_districts else None,
    crime_categories=selected_categories if selected_categories else None,
    date_range=date_range,
    risk_levels=selected_risks if selected_risks else None
)



# Navigation Logic
if menu == "📊 Analytics Center":
    st.markdown("<h1>📊 Crime Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Karnataka Police Intelligence Hub</h3>", unsafe_allow_html=True)
    
    # Calculate KPIs
    kpis = calculate_kpis(filtered_df)
    
    # KPI Grid using custom HTML for futuristic design
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Crimes Registered</div>
                <div class="metric-value">{kpis['total_crimes']:,}</div>
                <div class="metric-change">Statewide Ledger</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Active Investigations</div>
                <div class="metric-value">{kpis['active_cases']:,}</div>
                <div class="metric-change">{(kpis['active_cases']/kpis['total_crimes']*100):.1f}% Active Load</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Solved / Closed Cases</div>
                <div class="metric-value">{kpis['solved_cases']:,}</div>
                <div class="metric-change">{(kpis['solved_cases']/kpis['total_crimes']*100):.1f}% Solve Rate</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">High Risk Incidents</div>
                <div class="metric-value">{kpis['high_risk_cases']:,}</div>
                <div class="metric-change">Priority Focus</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    col5, col6, col7 = st.columns(3)
    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Repeat Offenders Tracked</div>
                <div class="metric-value">{kpis['repeat_offenders']}</div>
                <div class="metric-change">Active Threat Profiles</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with col6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Crime Growth Rate (60d)</div>
                <div class="metric-value {'negative' if kpis['growth_percentage'] > 0 else ''}">
                    {'+' if kpis['growth_percentage'] > 0 else ''}{kpis['growth_percentage']}%
                </div>
                <div class="metric-change">Recent Trend Delta</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with col7:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Crime Severity Index (CSI)</div>
                <div class="metric-value">{kpis['crime_severity_index']}/100</div>
                <div class="metric-change">Weighted Force Burden</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    
    # 2. Main Analytics Charts
    c_col1, c_col2 = st.columns(2)
    
    with c_col1:
        st.markdown("#### Category Distribution")
        cat_counts = filtered_df["Crime_Category"].value_counts().reset_index()
        cat_counts.columns = ["Crime Category", "Incidents"]
        fig_donut = px.pie(
            cat_counts, 
            names="Crime Category", 
            values="Incidents", 
            hole=0.4, 
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#ffffff")
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with c_col2:
        st.markdown("#### District Wise Crime Breakdown")
        dist_counts = filtered_df["District"].value_counts().reset_index()
        dist_counts.columns = ["District", "Incidents"]
        fig_dist = px.bar(
            dist_counts.head(10), 
            y="District", 
            x="Incidents", 
            orientation="h",
            color="Incidents",
            color_continuous_scale="Viridis"
        )
        fig_dist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#ffffff"),
            xaxis=dict(showgrid=True, gridcolor='#1E293B'),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    c_col3, c_col4 = st.columns(2)
    
    with c_col3:
        st.markdown("#### Time-Series Monthly Trends")
        filtered_df["YearMonth"] = filtered_df["Date"].dt.to_period("M").astype(str)
        trend_df = filtered_df.groupby("YearMonth").size().reset_index(name="Incidents")
        fig_trend = px.line(trend_df, x="YearMonth", y="Incidents", markers=True)
        fig_trend.update_traces(line_color="#00E5FF", line_width=3, marker=dict(size=8, color="#7209B7"))
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#ffffff"),
            xaxis=dict(showgrid=True, gridcolor='#1E293B'),
            yaxis=dict(showgrid=True, gridcolor='#1E293B')
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with c_col4:
        st.markdown("#### Investigation Status Distribution")
        status_counts = filtered_df["Investigation_Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig_status = px.bar(status_counts, x="Status", y="Count", color="Status", color_discrete_sequence=px.colors.qualitative.Safe)
        fig_status.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#ffffff"),
            yaxis=dict(showgrid=True, gridcolor='#1E293B')
        )
        st.plotly_chart(fig_status, use_container_width=True)

    # Crime Type Rankings
    st.markdown("#### Top Crime Types Ranking")
    type_counts = filtered_df["Crime_Type"].value_counts().reset_index()
    type_counts.columns = ["Crime Type", "Incidents"]
    fig_types = px.bar(
        type_counts.head(15), 
        x="Incidents", 
        y="Crime Type", 
        orientation="h",
        color="Incidents",
        color_continuous_scale="Plasma"
    )
    fig_types.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#ffffff"),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_types, use_container_width=True)
    
    st.markdown("---")
    
    # Feature 13: Natural Language Storytelling
    st.markdown("### 📖 Natural Language Crime Storytelling")
    story_briefs = generate_storytelling_briefs(filtered_df)
    for brief in story_briefs:
        st.markdown(
            f"""
            <div style='background-color: #0A0F24; border-left: 4px solid #7209B7; border-radius: 6px; padding: 15px; margin-bottom: 12px;'>
                <p style='margin: 0; font-size: 13.5px; line-height: 1.5; color: #E2E8F0;'>{brief}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

elif menu == "🗺️ Hotspot Intelligence":
    st.markdown("<h1>🗺️ Crime Hotspot Intelligence Map</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Interactive Spatial Analysis & Risk Clusters</h3>", unsafe_allow_html=True)
    
    # Coordinates of Karnataka Center
    center_lat = 14.8
    center_lon = 76.2
    
    # Initialize Folium Map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="cartodbpositron")
    
    # Filters applied to map
    map_df = filtered_df.dropna(subset=["Latitude", "Longitude"])
    
    # Checkbox filters for map views
    map_col1, map_col2 = st.columns([1, 4])
    with map_col1:
        st.markdown("#### Map Layers")
        show_heatmap = st.checkbox("Show Density Heatmap", value=True)
        show_markers = st.checkbox("Show Individual Incidents", value=False)
        show_district_scores = st.checkbox("Show District Risk Zones", value=True)
        
        # Display legend
        st.markdown(
            """
            <div style='border: 1px solid #1E293B; border-radius: 8px; padding: 10px; background-color: #070B19;'>
                <p style='margin: 0; font-size: 11px; font-weight: 700; color:#E2E8F0;'>RISK LEGEND</p>
                <div style='display: flex; align-items: center; margin-top: 5px;'>
                    <div style='width: 12px; height: 12px; border-radius: 50%; background-color: #DC2626; margin-right: 8px;'></div>
                    <span style='font-size: 11px;'>Red = High Risk</span>
                </div>
                <div style='display: flex; align-items: center; margin-top: 5px;'>
                    <div style='width: 12px; height: 12px; border-radius: 50%; background-color: #F59E0B; margin-right: 8px;'></div>
                    <span style='font-size: 11px;'>Orange = Medium Risk</span>
                </div>
                <div style='display: flex; align-items: center; margin-top: 5px;'>
                    <div style='width: 12px; height: 12px; border-radius: 50%; background-color: #10B981; margin-right: 8px;'></div>
                    <span style='font-size: 11px;'>Green = Low Risk</span>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with map_col2:
        # Add Heatmap Layer
        if show_heatmap and len(map_df) > 0:
            heat_data = map_df[["Latitude", "Longitude"]].values.tolist()
            HeatMap(heat_data, radius=15, blur=10).add_to(m)
            
        # Add District Risk Zone Circles
        if show_district_scores:
            leaderboard_df = get_district_leaderboard(df)
            from src.data_generator import KARNATAKA_DISTRICTS
            
            for _, row in leaderboard_df.iterrows():
                dist_name = row["District"]
                risk_score = row["Risk Score"]
                
                # Fetch centroid coordinates
                if dist_name in KARNATAKA_DISTRICTS:
                    coords = KARNATAKA_DISTRICTS[dist_name]["coords"]
                    
                    # Color coding based on risk score
                    if risk_score >= 65:
                        color = "#DC2626"  # Red
                    elif risk_score >= 35:
                        color = "#F59E0B"  # Orange
                    else:
                        color = "#10B981"  # Green
                        
                    folium.Circle(
                        location=coords,
                        radius=25000,  # 25km radius circle
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.25,
                        popup=folium.Popup(f"<b>District:</b> {dist_name}<br><b>Risk Score:</b> {risk_score}/100<br><b>Crime Count:</b> {row['Crime Count']}", max_width=200)
                    ).add_to(m)
        
        # Add Individual Incident Markers
        if show_markers and len(map_df) > 0:
            # Limit markers to 300 to avoid freezing streamlit-folium
            marker_sample = map_df.head(300)
            
            for _, row in marker_sample.iterrows():
                risk = row["Risk_Level"]
                color = "#10B981"  # Green
                if risk == "High":
                    color = "#DC2626"
                elif risk == "Medium":
                    color = "#F59E0B"
                    
                popup_content = f"""
                <b>FIR ID:</b> {row['FIR_ID']}<br>
                <b>Category:</b> {row['Crime_Category']}<br>
                <b>Type:</b> {row['Crime_Type']}<br>
                <b>District:</b> {row['District']}<br>
                <b>Severity:</b> {row['Crime_Severity']}<br>
                <b>Risk Level:</b> {row['Risk_Level']}<br>
                <b>Date:</b> {row['Date'].strftime('%d-%m-%Y')}
                """
                
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    popup=folium.Popup(popup_content, max_width=250)
                ).add_to(m)
                
        # Render the map
        st_folium(m, width="100%", height=600)

elif menu == "🔮 Predictive Intelligence":
    st.markdown("<h1>🔮 AI Crime Prediction Engine & Risk Scoring</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Predictive Policing Framework & Explainable AI</h3>", unsafe_allow_html=True)
    
    tab_vol, tab_risk, tab_board = st.tabs(["📊 Future Volume Forecast", "🛡️ Crime Risk Scoring System", "🏆 District Leaderboard"])
    
    with tab_vol:
        st.markdown("#### Forecast Future Monthly Crime Volume")
        st.markdown("Predict how many crimes are expected to occur in a given district next month based on historical trends.")
        
        # Form inputs for forecasting
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            f_district = st.selectbox("Target District", districts_list, key="f_dist")
            f_category = st.selectbox("Crime Category", crime_cats_list, key="f_cat")
            f_month = st.slider("Month of Prediction", 1, 12, int(datetime.now().month))
            f_year = st.selectbox("Year of Prediction", [2026, 2027], index=0)
            
        with v_col2:
            # Pull historical average monthly count for district/category to seed default inputs
            hist_df = df[(df["District"] == f_district) & (df["Crime_Category"] == f_category)]
            hist_df['YearMonth'] = hist_df['Date'].dt.to_period("M")
            monthly_counts = hist_df.groupby('YearMonth').size()
            avg_count = float(monthly_counts.mean()) if len(monthly_counts) > 0 else 10.0
            
            f_lag1 = st.number_input("Crime Volume 1 Month Prior (Lag 1)", min_value=0.0, value=avg_count)
            f_lag2 = st.number_input("Crime Volume 2 Months Prior (Lag 2)", min_value=0.0, value=avg_count)
            
            predict_vol_btn = st.button("Generate Volume Forecast")
            
        if predict_vol_btn:
            pred_vol = ml_engine.predict_future_volume(f_district, f_category, f_month, f_year, f_lag1, f_lag2)
            
            st.markdown("---")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown(
                    f"""
                    <div class="metric-card" style='border-color: #7209B7;'>
                        <div class="metric-title">Forecasted Crime Count ({f_district})</div>
                        <div class="metric-value">{pred_vol} Incidents</div>
                        <div class="metric-change" style='color: #00E5FF;'>For Month {f_month}/{f_year}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            with p_col2:
                st.markdown("##### Prediction Engine Confidence Metrics")
                st.markdown("- **Model Type**: Gradient Boosting Regressor")
                st.markdown("- **Historical Average Monthly Volume**: %.1f" % avg_count)
                st.markdown("- **Forecast Target Category**: %s" % f_category)
                trend_direction = "INCREASING" if pred_vol > avg_count else "DECREASING"
                st.markdown(f"- **Implied Volume Trend**: **{trend_direction}** vs historical baseline")
                
    with tab_risk:
        st.markdown("#### AI Incident Risk Scoring System")
        st.markdown("Input information about a hypothetical or real FIR to calculate its AI risk classification, complete with explainable justifications.")
        
        # Incident parameters form
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            input_cat = st.selectbox("Crime Category", crime_cats_list, key="rc_cat")
            
            # Filter type based on selected category
            from src.data_generator import CRIME_MAPPING
            available_types = CRIME_MAPPING.get(input_cat, ["Other"])
            input_type = st.selectbox("Crime Type", available_types, key="rc_type")
            
            input_district = st.selectbox("Incident District", districts_list, key="rc_dist")
            
            # Filter stations based on district
            from src.data_generator import KARNATAKA_DISTRICTS
            available_stations = KARNATAKA_DISTRICTS[input_district]["stations"]
            input_station = st.selectbox("Police Station", available_stations, key="rc_station")
            
            input_severity = st.selectbox("Crime Severity", ["Low", "Medium", "High"], index=1)
            
        with r_col2:
            input_prev_offenses = st.slider("Suspect Previous Offenses", 0, 10, value=1)
            input_se_index = st.slider("District Socio-Economic Index", 1, 100, value=65)
            input_loss = st.number_input("Financial Loss Involved (Rs.)", min_value=0.0, value=25000.0)
            
            # Demographics
            input_v_age = st.slider("Victim Age", 5, 95, 35)
            input_v_gender = st.selectbox("Victim Gender", ["Male", "Female", "Other"])
            input_s_age = st.slider("Suspect Age", 15, 85, 28)
            input_s_gender = st.selectbox("Suspect Gender", ["Male", "Female", "Other"])
            
            score_btn = st.button("Analyze Incident Risk Score")
            
        if score_btn:
            input_dict = {
                "Crime_Category": input_cat,
                "Crime_Type": input_type,
                "District": input_district,
                "Police_Station": input_station,
                "Crime_Severity": input_severity,
                "Previous_Offenses": input_prev_offenses,
                "Socio_Economic_Index": input_se_index,
                "Financial_Loss": input_loss,
                "Victim_Age": input_v_age,
                "Victim_Gender": input_v_gender,
                "Suspect_Age": input_s_age,
                "Suspect_Gender": input_s_gender
            }
            
            risk_result = ml_engine.predict_risk(input_dict)
            
            st.markdown("---")
            # Displays output risk gauges and explainability
            g_col1, g_col2 = st.columns([2, 3])
            
            with g_col1:
                # Gauge representation of risk level and score
                pred_label = risk_result["prediction"]
                conf = risk_result["confidence"]
                
                # Determine color
                gauge_color = "#10B981"  # green
                if pred_label == "High":
                    gauge_color = "#DC2626"
                elif pred_label == "Medium":
                    gauge_color = "#F59E0B"
                    
                # Calculate numeric risk score out of 100 based on probabilities
                high_prob = risk_result["probabilities"].get("High", 0.0)
                med_prob = risk_result["probabilities"].get("Medium", 0.0)
                risk_score_gauge = int((high_prob * 100) + (med_prob * 50))
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_score_gauge,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"AI RISK LEVEL: {pred_label.upper()}", 'font': {'size': 18, 'color': gauge_color}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickcolor': "#ffffff"},
                        'bar': {'color': gauge_color},
                        'bgcolor': "#0A0F24",
                        'borderwidth': 2,
                        'bordercolor': "#1E293B",
                        'steps': [
                            {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.15)'},
                            {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.15)'},
                            {'range': [65, 100], 'color': 'rgba(220, 38, 38, 0.15)'}
                        ]
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    font=dict(color="#ffffff"),
                    height=280,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with g_col2:
                # Feature contributions horizontal bar chart
                st.markdown("##### Feature Contributions to Score (Local Explanation)")
                feat_w = risk_result["weights"]
                keys = list(feat_w.keys())
                vals = list(feat_w.values())
                
                fig_contrib = px.bar(
                    x=vals, y=keys, 
                    orientation='h',
                    color=vals,
                    color_continuous_scale="RdYlGn_r",
                    color_continuous_midpoint=0
                )
                fig_contrib.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#ffffff"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=260,
                    xaxis=dict(showgrid=True, gridcolor='#1E293B', title="Risk Contribution Force"),
                    yaxis=dict(showgrid=False, title="")
                )
                st.plotly_chart(fig_contrib, use_container_width=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            # Explainable AI output
            st.markdown("##### Explainable AI (XAI) - Why was this prediction made?")
            for text in risk_result["explanations"]:
                st.info(text)
                
            # Global model feature importance
            st.markdown("##### Global Model Feature Importance (All Training Data)")
            importances = risk_result["importances"]
            features = risk_result["features"]
            # Sort importances
            sorted_idx = np.argsort(importances)
            sorted_feats = [features[i] for i in sorted_idx]
            sorted_imps = [importances[i] for i in sorted_idx]
            
            fig_glob = px.bar(
                x=sorted_imps, y=sorted_feats, 
                orientation='h',
                labels={'x': 'Relative Importance', 'y': 'Feature Name'}
            )
            fig_glob.update_traces(marker_color='#00E5FF')
            fig_glob.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#ffffff"),
                xaxis=dict(showgrid=True, gridcolor='#1E293B'),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_glob, use_container_width=True)

    with tab_board:
        st.markdown("#### District Leaderboard")
        st.markdown("Ranking top 10 districts in Karnataka state across critical indicators.")
        
        # Calculate district leaderboard
        leaderboard_df = get_district_leaderboard(filtered_df)
        
        # Dropdown selection for metric sorting
        sort_metric = st.selectbox(
            "Rank districts by:",
            ["Crime Count", "Risk Score", "Crime Growth Rate (%)", "Severity Index"]
        )
        
        sorted_leaderboard = leaderboard_df.sort_values(by=sort_metric, ascending=False).head(10).reset_index(drop=True)
        sorted_leaderboard.index = sorted_leaderboard.index + 1
        
        # Visual table display
        st.dataframe(
            sorted_leaderboard, 
            use_container_width=True,
            column_config={
                "District": st.column_config.TextColumn("District Name"),
                "Crime Count": st.column_config.NumberColumn("Total Crimes", format="%d"),
                "Risk Score": st.column_config.ProgressColumn("Risk Score (0-100)", min_value=0, max_value=100, format="%.1f"),
                "Crime Growth Rate (%)": st.column_config.NumberColumn("Growth Rate", format="%.2f%%"),
                "Severity Index": st.column_config.ProgressColumn("Severity Index", min_value=0, max_value=100, format="%.1f")
            }
        )

elif menu == "🕸️ Criminal Network Graph":
    st.markdown("<h1>🕸️ Criminal Network Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Repeat Offenders, Organized Crime, & Hidden Linkages</h3>", unsafe_allow_html=True)
    
    # Filter selection for network
    net_district = st.selectbox("Select District Focus (Optional)", ["All Districts"] + districts_list, index=0)
    limit_nodes = st.slider("Node Count Limitation (for layout clarity)", 20, 200, value=75)
    
    district_param = None if net_district == "All Districts" else net_district
    
    # Build Network Graph
    with st.spinner("Processing relationship metrics..."):
        G, node_types, node_details, deg_cent, bet_cent, net_stats = build_crime_network(
            filtered_df, district=district_param, limit=limit_nodes
        )
        
    net_col1, net_col2 = st.columns([3, 1])
    
    with net_col1:
        st.markdown("#### Interactive Relationship Graph")
        fig_net = visualize_network_plotly(G, node_types, node_details, deg_cent)
        fig_net.update_layout(height=650)
        st.plotly_chart(fig_net, use_container_width=True)
        
    with net_col2:
        st.markdown("#### Network Overview")
        st.markdown(
            f"""
            <div style='background-color: #070B19; border: 1px solid #1E293B; border-radius: 8px; padding: 15px;'>
                <p style='margin: 0; color: #94A3B8; font-size: 11px;'>TOTAL NETWORK NODES</p>
                <h3 style='margin: 4px 0 0 0; color: #00E5FF !important;'>{net_stats['num_nodes']}</h3>
                <p style='margin: 10px 0 0 0; color: #94A3B8; font-size: 11px;'>TOTAL LINKS / EDGES</p>
                <h3 style='margin: 4px 0 0 0; color: #00E5FF !important;'>{net_stats['num_edges']}</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        st.markdown("#### High Threat Targets<br><small>Ranked by Degree Centrality</small>", unsafe_allow_html=True)
        
        for idx, (sus_id, centrality) in enumerate(net_stats["top_suspects"], 1):
            # Find suspect details in dataframe
            sus_info = filtered_df[filtered_df["Suspect_ID"] == sus_id].iloc[0]
            st.markdown(
                f"""
                <div style='background-color: #0A0F24; border: 1px solid #1E293B; border-radius: 6px; padding: 10px; margin-bottom: 8px;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <span style='font-weight: 700; color: #E63946;'>#{idx} {sus_id}</span>
                        <span style='font-size: 10px; color: #64748B;'>Cent: {centrality:.2f}</span>
                    </div>
                    <p style='margin: 4px 0 0 0; font-size: 11px; color: #94A3B8;'>
                        Age: {sus_info['Suspect_Age']} | Gender: {sus_info['Suspect_Gender']}<br>
                        Prior Offenses: <b style='color:#ffffff'>{sus_info['Previous_Offenses']}</b>
                    </p>
                </div>
                """, 
                unsafe_allow_html=True
            )

elif menu == "🚨 Anomalies & Alert Feed":
    st.markdown("<h1>🚨 AI Anomaly Detection & Alerts</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Statistical Outliers & Rapid Threat Centers</h3>", unsafe_allow_html=True)
    
    # Feature 7: Anomaly detection
    st.markdown("#### Statistical Outliers & Activity Spikes")
    anomalies = detect_anomalies(filtered_df)
    
    if len(anomalies) == 0:
        st.success("No statistical anomalies detected in the selected data segment.")
    else:
        for anom in anomalies:
            severity_class = "medium" if anom["severity"] == "Medium" else "high"
            border_color = "#F59E0B" if anom["severity"] == "Medium" else "#EF4444"
            st.markdown(
                f"""
                <div class="alert-card {severity_class}" style='border-left-color: {border_color};'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span class="alert-title">[{anom['type'].upper()}] {anom['title']}</span>
                        <span style='background-color: {border_color}; color: #000; font-size: 9px; font-weight:800; padding: 2px 6px; border-radius: 4px;'>{anom['severity'].upper()} PRIORITY</span>
                    </div>
                    <p class="alert-desc" style='margin-top: 6px;'>{anom['description']}</p>
                    <div class="alert-meta">Detected: {anom['timestamp']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    st.markdown("---")
    
    # Feature 12: Emerging Crime Alert Center
    st.markdown("#### Emerging Crime Alert Center")
    st.markdown("Real-time feed showing active threats, rising vectors, and recidivism notices.")
    
    alerts = generate_alerts(filtered_df)
    
    col_a1, col_a2 = st.columns(2)
    for i, alert in enumerate(alerts):
        # alternate between columns
        target_col = col_a1 if i % 2 == 0 else col_a2
        
        badge_color = "#E63946" if alert["severity"] == "High" else "#FFB703"
        
        with target_col:
            st.markdown(
                f"""
                <div style='background-color: #0A0F24; border: 1px solid #1E293B; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.3);'>
                    <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1E293B; padding-bottom: 8px; margin-bottom: 10px;'>
                        <span style='font-size: 11px; font-weight: 700; color: #00E5FF; text-transform: uppercase;'>{alert['category']}</span>
                        <span style='background-color: {badge_color}; color: #050811; font-size: 9px; font-weight:800; padding: 2px 6px; border-radius: 3px;'>{alert['severity'].upper()}</span>
                    </div>
                    <h5 style='margin: 0; font-size: 14px; color: #ffffff !important;'>{alert['title']}</h5>
                    <p style='margin: 8px 0 0 0; font-size: 12px; color: #94A3B8; line-height: 1.4;'>{alert['description']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

elif menu == "📄 Intelligence Report":
    st.markdown("<h1>📄 Crime Intelligence Report Generator</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Downloadable Briefings & Natural Language Recommendations</h3>", unsafe_allow_html=True)
    
    # Feature 10: Strategic Recommendations
    st.markdown("#### Strategic Recommendations")
    recs = generate_recommendations(filtered_df)
    
    for idx, rec in enumerate(recs, 1):
        st.markdown(
            f"""
            <div style='background-color: #0A0F24; border-left: 4px solid #00B4D8; border-radius: 6px; padding: 15px; margin-bottom: 12px;'>
                <h5 style='margin: 0; font-size: 14.5px; color: #00E5FF !important;'>Recommendation {idx}</h5>
                <p style='margin: 5px 0 0 0; font-size: 12.5px; color: #E2E8F0; line-height: 1.45;'>{rec}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    st.markdown("#### Export Briefing Document")
    st.markdown("Compile current filtered statistics, leaderboards, predictions, and recommendations into a formatted PDF file.")
    
    generate_btn = st.button("Compile & Generate PDF Report")
    
    if generate_btn:
        # Create necessary inputs for PDF
        kpis = calculate_kpis(filtered_df)
        leaderboard_df = get_district_leaderboard(filtered_df)
        
        # Mock some future volume forecasts for the PDF
        top_districts = leaderboard_df.sort_values(by="Crime Count", ascending=False).head(3)["District"].tolist()
        pdf_predictions = []
        for dist in top_districts:
            for cat in ["Property Crimes", "Cyber Crimes"]:
                hist_avg = len(filtered_df[(filtered_df["District"] == dist) & (filtered_df["Crime_Category"] == cat)]) / 30.0
                hist_avg = max(1.0, hist_avg)
                pred_val = ml_engine.predict_future_volume(dist, cat, 7, 2026, hist_avg, hist_avg)
                pdf_predictions.append({
                    "district": dist,
                    "category": cat,
                    "current": hist_avg,
                    "predicted": pred_val
                })
                
        output_pdf_path = "reports/crime_report.pdf"
        
        with st.spinner("Compiling PDF document layout..."):
            generate_pdf_report(
                filtered_df,
                kpis,
                leaderboard_df,
                pdf_predictions,
                recs,
                output_path=output_pdf_path
            )
            
        st.success("PDF Report generated successfully!")
        
        # Read file for downloading
        with open(output_pdf_path, "rb") as f:
            pdf_bytes = f.read()
            
        st.download_button(
            label="⬇️ Download PDF Briefing",
            data=pdf_bytes,
            file_name=f"CrimeVision_Intelligence_Report_{datetime.now().strftime('%d_%m_%Y')}.pdf",
            mime="application/pdf"
        )
