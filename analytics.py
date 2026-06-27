import pandas as pd
import numpy as np

def load_data(csv_path="data/synthetic_crime_data.csv"):
    """Loads the crime dataset from the CSV file."""
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def calculate_kpis(df):
    """
    Computes key performance indicators:
    - Total Crimes
    - Active Cases (Under Investigation)
    - Solved Cases (Chargesheeted, Closed, Arrested)
    - High Risk Cases (Risk_Level == High)
    - Repeat Offenders (Suspects who have Previous_Offenses > 0 or occur > 1 time)
    - Crime Growth Percentage (Comparing last 30 days to previous 30 days)
    - Crime Severity Index (Average severity score 0-100)
    """
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
    total_crimes = len(df)
    
    # Active Cases
    active_statuses = ["Under Investigation"]
    active_cases = len(df[df["Investigation_Status"].isin(active_statuses)])
    
    # Solved Cases
    solved_statuses = ["Chargesheeted", "Closed", "Arrested"]
    solved_cases = len(df[df["Investigation_Status"].isin(solved_statuses)])
    
    # High Risk Cases
    high_risk_cases = len(df[df["Risk_Level"] == "High"])
    
    # Repeat Offenders
    # Count suspects who commit more than 1 crime in this dataset,
    # or who have previous offenses > 0.
    suspect_counts = df["Suspect_ID"].value_counts()
    repeat_suspect_ids_in_data = suspect_counts[suspect_counts > 1].index.tolist()
    
    # Combined with declared previous offenses
    repeat_offenders_df = df[(df["Suspect_ID"].isin(repeat_suspect_ids_in_data)) | (df["Previous_Offenses"] > 0)]
    repeat_offenders = repeat_offenders_df["Suspect_ID"].nunique()
    
    # Crime Growth Percentage (last 60 days vs prior 60 days to ensure enough volume)
    max_date = df['Date'].max()
    sixty_days_ago = max_date - pd.Timedelta(days=60)
    one_twenty_days_ago = max_date - pd.Timedelta(days=120)
    
    recent_period = df[(df['Date'] >= sixty_days_ago) & (df['Date'] <= max_date)]
    prior_period = df[(df['Date'] >= one_twenty_days_ago) & (df['Date'] < sixty_days_ago)]
    
    recent_count = len(recent_period)
    prior_count = len(prior_period)
    
    if prior_count > 0:
        growth_percentage = ((recent_count - prior_count) / prior_count) * 100
    else:
        growth_percentage = 0.0
        
    # Crime Severity Index (CSI)
    # Severity Mapping: High = 100, Medium = 50, Low = 15
    severity_map = {"High": 100, "Medium": 50, "Low": 15}
    csi = df["Crime_Severity"].map(severity_map).mean()
    if pd.isna(csi):
        csi = 0.0
        
    return {
        "total_crimes": total_crimes,
        "active_cases": active_cases,
        "solved_cases": solved_cases,
        "high_risk_cases": high_risk_cases,
        "repeat_offenders": repeat_offenders,
        "growth_percentage": round(growth_percentage, 2),
        "crime_severity_index": round(csi, 2)
    }

def get_filtered_data(df, districts=None, crime_categories=None, date_range=None, risk_levels=None):
    """Filters the dataframe dynamically based on user selections."""
    filtered_df = df.copy()
    
    if districts:
        filtered_df = filtered_df[filtered_df["District"].isin(districts)]
        
    if crime_categories:
        filtered_df = filtered_df[filtered_df["Crime_Category"].isin(crime_categories)]
        
    if date_range and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = filtered_df[(filtered_df["Date"] >= start_date) & (filtered_df["Date"] <= end_date)]
        
    if risk_levels:
        filtered_df = filtered_df[filtered_df["Risk_Level"].isin(risk_levels)]
        
    return filtered_df

def get_district_leaderboard(df):
    """
    Computes rankings for districts based on:
    - Crime Count
    - Average Risk Score (synthesized)
    - Crime Growth Rate (last 60 days vs prior 60 days)
    - Severity Index (CSI)
    """
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
    districts = df["District"].unique()
    leaderboard_data = []
    
    max_date = df['Date'].max()
    sixty_days_ago = max_date - pd.Timedelta(days=60)
    one_twenty_days_ago = max_date - pd.Timedelta(days=120)
    
    severity_map = {"High": 100, "Medium": 50, "Low": 15}
    
    for district in districts:
        dist_df = df[df["District"] == district]
        crime_count = len(dist_df)
        
        # Risk levels map
        risk_map = {"High": 100, "Medium": 50, "Low": 10}
        risk_score = dist_df["Risk_Level"].map(risk_map).mean()
        if pd.isna(risk_score):
            risk_score = 0.0
            
        severity_index = dist_df["Crime_Severity"].map(severity_map).mean()
        if pd.isna(severity_index):
            severity_index = 0.0
            
        # Growth Rate for District
        recent_count = len(dist_df[(dist_df['Date'] >= sixty_days_ago) & (dist_df['Date'] <= max_date)])
        prior_count = len(dist_df[(dist_df['Date'] >= one_twenty_days_ago) & (dist_df['Date'] < sixty_days_ago)])
        
        if prior_count > 0:
            growth_rate = ((recent_count - prior_count) / prior_count) * 100
        else:
            growth_rate = 0.0
            
        leaderboard_data.append({
            "District": district,
            "Crime Count": crime_count,
            "Risk Score": round(risk_score, 1),
            "Crime Growth Rate (%)": round(growth_rate, 2),
            "Severity Index": round(severity_index, 1)
        })
        
    return pd.DataFrame(leaderboard_data)
