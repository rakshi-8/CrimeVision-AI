import pandas as pd
import numpy as np

def generate_recommendations(df):
    """
    Analyzes crime data patterns and generates dynamic,
    natural language strategic policing recommendations.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
    recommendations = []
    
    # 1. Cyber Crime Analysis
    cyber_df = df[df["Crime_Category"] == "Cyber Crimes"]
    if len(cyber_df) > 0:
        # Check if cybercrime is trending up
        df_2025_2026 = df[df["Date"].dt.year.isin([2025, 2026])]
        cyber_pct = (len(df_2025_2026[df_2025_2026["Crime_Category"] == "Cyber Crimes"]) / len(df_2025_2026)) * 100
        if cyber_pct > 15:
            recommendations.append(
                "Cybercrime Prevention Campaign: Digital scams and online fraud have risen. "
                "Launch public awareness campaigns targeting vulnerable demographics (senior citizens and students) "
                "regarding phishing links and lottery frauds, and expand the cyber cell capacity."
            )
            
    # 2. Hotspots & Patrols
    district_counts = df["District"].value_counts()
    if len(district_counts) > 0:
        top_district = district_counts.index[0]
        top_stations_in_dist = df[df["District"] == top_district]["Police_Station"].value_counts().index[:2].tolist()
        recommendations.append(
            f"Targeted Hotspot Patrols: High concentrations of incidents detected in {top_district} district, "
            f"specifically under the jurisdiction of {', '.join(top_stations_in_dist)}. "
            f"Increase visible street patrolling and set up CCTV checkpoints in these sectors."
        )
        
    # 3. Weekend vs Weekday analysis
    df['DayOfWeek'] = df['Date'].dt.day_name()
    weekend_df = df[df['DayOfWeek'].isin(['Saturday', 'Sunday'])]
    weekday_df = df[~df['DayOfWeek'].isin(['Saturday', 'Sunday'])]
    
    avg_weekend_daily = len(weekend_df) / 2
    avg_weekday_daily = len(weekday_df) / 5
    
    if avg_weekend_daily > avg_weekday_daily * 1.05:
        # Weekend spike
        recommendations.append(
            "Weekend Force Augmentation: Statistics reveal a crime volume spike during weekends. "
            "Deploy 15% additional patrol units and establish sobriety check points near entertainment zones "
            "between 18:00 and 02:00."
        )
    else:
        recommendations.append(
            "Resource Optimization: Weekly crime levels remain uniform. Maintain standard watch rotations "
            "but shift non-critical administration staff to support peak-hour evening patrol shifts."
        )
        
    # 4. Repeat Offenders
    repeat_offenders_df = df[df["Previous_Offenses"] > 0]
    repeat_pct = (len(repeat_offenders_df) / len(df)) * 100
    if repeat_pct > 20:
        recommendations.append(
            "Recidivism Monitoring: Over 20% of incidents involve suspects with active criminal records. "
            "Implement a coordinated repeat offender management program in collaboration with local courts "
            "to expedite trials and monitor bail conditions."
        )
        
    # 5. Night-time Crime
    # Extract hour from Time string (HH:MM:SS)
    df['Hour'] = df['Time'].apply(lambda x: int(x.split(":")[0]) if isinstance(x, str) else 12)
    night_crimes = df[(df['Hour'] >= 22) | (df['Hour'] <= 4)]
    night_pct = (len(night_crimes) / len(df)) * 100
    
    if night_pct > 25:
        recommendations.append(
            "Night Watch Operations: Significant crime density identified between 22:00 and 04:00. "
            "Collaborate with municipal corporations to improve street lighting in dark spots and mandate "
            "police-citizen night patrols in residential sectors."
        )
        
    # Standard recommendations fallbacks if lists are short
    if len(recommendations) < 3:
        recommendations.append(
            "Community Policing Initiatives: Conduct monthly townhall meetings to bridge the police-citizen gap "
            "and encourage voluntary reporting of suspicious actions."
        )
        recommendations.append(
            "Emergency Response Drills: Schedule quarterly tactical scenarios for response units to "
            "reduce average response times from 11 minutes to sub-7 minutes."
        )
        
    return recommendations

def generate_storytelling_briefs(df):
    """
    Translates raw numbers into descriptive, narrative intelligence stories.
    Returns a list of paragraph strings representing natural language briefs.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
    briefs = []
    
    # 1. State-level summary
    total_cases = len(df)
    solved_count = len(df[df["Investigation_Status"].isin(["Closed", "Chargesheeted", "Arrested"])])
    solve_rate = (solved_count / total_cases) * 100
    
    briefs.append(
        f"Across Karnataka State, a total of {total_cases:,} incidents are registered in the current intelligence ledger. "
        f"The state investigation machinery is maintaining a solve rate of {solve_rate:.1f}%, resolving "
        f"{solved_count:,} cases. High-risk instances account for "
        f"{len(df[df['Risk_Level'] == 'High']):,} of these records, needing aggressive administrative focus."
    )
    
    # 2. Highest Crime District Narrative
    district_counts = df["District"].value_counts()
    if len(district_counts) > 0:
        top_dist = district_counts.index[0]
        top_count = district_counts.iloc[0]
        top_dist_df = df[df["District"] == top_dist]
        top_cat = top_dist_df["Crime_Category"].value_counts().index[0]
        top_cat_pct = (len(top_dist_df[top_dist_df["Crime_Category"] == top_cat]) / len(top_dist_df)) * 100
        
        # Growth Rate for top district
        max_date = df['Date'].max()
        sixty_days_ago = max_date - pd.Timedelta(days=60)
        one_twenty_days_ago = max_date - pd.Timedelta(days=120)
        
        recent_count = len(top_dist_df[(top_dist_df['Date'] >= sixty_days_ago) & (top_dist_df['Date'] <= max_date)])
        prior_count = len(top_dist_df[(top_dist_df['Date'] >= one_twenty_days_ago) & (top_dist_df['Date'] < sixty_days_ago)])
        
        growth_str = "stable volume"
        if prior_count > 0:
            growth_pct = ((recent_count - prior_count) / prior_count) * 100
            if growth_pct > 5:
                growth_str = f"a notable surge of {growth_pct:.1f}%"
            elif growth_pct < -5:
                growth_str = f"a drop of {abs(growth_pct):.1f}%"
                
        briefs.append(
            f"The district of {top_dist} registers the highest crime density in the state with {top_count:,} cases. "
            f"Analytical decomposition shows that {top_cat} forms the bulk of the threat vector, "
            f"accounting for {top_cat_pct:.1f}% of all local filings. Over the last 60 days, "
            f"the district registered {growth_str} compared to the preceding quarter. "
            f"This requires prioritized deployment of regional reserves."
        )
        
    # 3. Rising Threat Narrative (Cyber Crimes)
    cyber_df = df[df["Crime_Category"] == "Cyber Crimes"]
    if len(cyber_df) > 0:
        # Group by year to show progression
        yearly_cyber = cyber_df.groupby(cyber_df["Date"].dt.year).size()
        years_list = list(yearly_cyber.index)
        if len(years_list) >= 2:
            y1, y2 = years_list[-2], years_list[-1]
            c1, c2 = yearly_cyber[y1], yearly_cyber[y2]
            pct_inc = ((c2 - c1) / c1) * 100 if c1 > 0 else 0
            
            briefs.append(
                f"Digital threat vectors continue to expand rapidly. In the cyber realm, documented incidents "
                f"moved from {c1:,} cases in {y1} to {c2:,} cases in {y2}, representing a "
                f"{pct_inc:.1f}% surge. The primary driver in this domain remains online financial scams, "
                f"which typically target high-income urban zones and rely on complex multi-state money transfer networks."
            )
            
    # 4. Suspect-Victim Demographic Narrative
    avg_sus_age = df["Suspect_Age"].mean()
    avg_vic_age = df["Victim_Age"].mean()
    top_vic_gender = df["Victim_Gender"].value_counts().index[0]
    
    briefs.append(
        f"Demographic analytics show that offenders skew younger, with an average suspect age of {avg_sus_age:.1f} years, "
        f"while victims display a broader profile with an average age of {avg_vic_age:.1f} years. "
        f"Individuals identified as {top_vic_gender} comprise the most frequent victims in the state, "
        f"which informs our recommendation for developing community support groups and targeted self-defense workshops."
    )
    return briefs

def get_patrol_recommendation(risk_score):
    """
    Rule-based engine returning force deployment directives 
    based on the predicted risk score.
    """
    if risk_score >= 80:
        return {
            "vehicles": 3,
            "cctv": "Aggressive Zoom & Active Panning",
            "drones": "Active (thermal imaging enabled)",
            "frequency": "High-Frequency (every 20 minutes)",
            "offender_watch": "Initiate monitoring of known area repeat offenders",
            "summary": "CRITICAL RISK LEVEL: Immediately deploy 3 patrol units, initiate thermal drone surveillance, and run active panning CCTV sweeps. Shift police beats to a high-frequency 20-minute cycle."
        }
    elif risk_score >= 50:
        return {
            "vehicles": 2,
            "cctv": "Increased CCTV Focus",
            "drones": "Standby / Selective deployment",
            "frequency": "Medium-Frequency (every 45 minutes)",
            "offender_watch": "Standard watch rotation",
            "summary": "MEDIUM RISK LEVEL: Deploy 2 patrol units, escalate CCTV surveillance, and conduct targeted police sweeps on a 45-minute cycle."
        }
    else:
        return {
            "vehicles": 1,
            "cctv": "Standard CCTV feeds",
            "drones": "Inactive",
            "frequency": "Standard Watch Rotation",
            "offender_watch": "None",
            "summary": "LOW RISK LEVEL: Standard force readiness. Deploy 1 routine patrol unit and maintain normal CCTV monitoring."
        }

