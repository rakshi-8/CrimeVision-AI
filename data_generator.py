import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Karnataka Districts, Centroids, and Police Stations
KARNATAKA_DISTRICTS = {
    "Bengaluru Urban": {
        "coords": (12.9716, 77.5946),
        "stations": ["Koramangala PS", "Indiranagar PS", "Whitefield PS", "Cubbon Park PS", "Jayanagar PS"],
        "crime_weight": 0.35,  # Higher weight for state capital
        "socio_economic_range": (30, 85)
    },
    "Mysuru": {
        "coords": (12.2958, 76.6394),
        "stations": ["Lashkar PS", "Devaraja PS", "Krishnaraja PS", "Vidyaranyapuram PS"],
        "crime_weight": 0.12,
        "socio_economic_range": (40, 75)
    },
    "Hubballi-Dharwad": {
        "coords": (15.3647, 75.1240),
        "stations": ["Gokul Road PS", "Suburban PS", "Town PS", "Keshwapur PS"],
        "crime_weight": 0.10,
        "socio_economic_range": (35, 65)
    },
    "Mangaluru": {
        "coords": (12.9141, 74.8560),
        "stations": ["Pandeshwar PS", "Kadri PS", "Urwa PS", "Barkur PS"],
        "crime_weight": 0.08,
        "socio_economic_range": (50, 85)
    },
    "Belagavi": {
        "coords": (15.8497, 74.4977),
        "stations": ["Market PS", "Khade Bazar PS", "Camp PS"],
        "crime_weight": 0.07,
        "socio_economic_range": (30, 60)
    },
    "Kalaburagi": {
        "coords": (17.3297, 76.8343),
        "stations": ["Chowk PS", "Station Bazar PS", "Raghavendra Nagar PS"],
        "crime_weight": 0.08,
        "socio_economic_range": (20, 50)
    },
    "Davanagere": {
        "coords": (14.4644, 75.9218),
        "stations": ["Extension PS", "Gandhinagar PS", "KTJ Nagar PS"],
        "crime_weight": 0.04,
        "socio_economic_range": (35, 60)
    },
    "Ballari": {
        "coords": (15.1394, 76.9214),
        "stations": ["Brucepet PS", "Gandhinagar PS", "Cowlbazar PS"],
        "crime_weight": 0.06,
        "socio_economic_range": (25, 55)
    },
    "Shivamogga": {
        "coords": (13.9299, 75.5681),
        "stations": ["Doddapet PS", "Kote PS", "Tunga Nagar PS"],
        "crime_weight": 0.03,
        "socio_economic_range": (40, 70)
    },
    "Tumakuru": {
        "coords": (13.3379, 77.1173),
        "stations": ["Kyathsandra PS", "Town PS", "New Extension PS"],
        "crime_weight": 0.03,
        "socio_economic_range": (35, 65)
    },
    "Hassan": {
        "coords": (13.0068, 76.1026),
        "stations": ["Hassan Central PS", "Hassan Rural PS", "Pension Lane PS"],
        "crime_weight": 0.02,
        "socio_economic_range": (40, 70)
    },
    "Chikkamagaluru": {
        "coords": (13.3161, 75.7720),
        "stations": ["Chikkamagaluru Town PS", "Rural PS", "Basavanahalli PS"],
        "crime_weight": 0.02,
        "socio_economic_range": (45, 75)
    }
}

# Crime Categories and their constituent Types
CRIME_MAPPING = {
    "Cyber Crimes": ["Online Fraud", "Identity Theft", "Phishing", "Cyber Bullying", "Hacking"],
    "Property Crimes": ["Theft", "House Breaking", "Vehicle Theft", "Robbery"],
    "Violent Crimes": ["Assault", "Murder", "Kidnapping", "Extortion"],
    "Economic Crimes": ["Financial Fraud", "Tax Evasion", "Bribery", "Money Laundering"],
    "Narcotics": ["Drug Trafficking", "Drug Possession", "Substance Abuse"],
    "Crimes Against Women": ["Harassment", "Domestic Violence", "Stalking"]
}

# Investigation Status options with weights
INVESTIGATION_STATUSES = ["Under Investigation", "Chargesheeted", "Closed", "Arrested", "Untraced"]
STATUS_WEIGHTS = [0.4, 0.35, 0.1, 0.1, 0.05]

# Suspect profile pool
NUM_SUSPECTS = 600
SUSPECT_POOL = []
for i in range(1, NUM_SUSPECTS + 1):
    sus_gender = random.choice(["Male", "Female", "Other"])
    # 85% male suspects for typical historical ratios, but with some variation
    if random.random() < 0.85:
        sus_gender = "Male"
    else:
        sus_gender = "Female" if random.random() < 0.9 else "Other"
        
    sus_age = int(np.clip(np.random.normal(32, 10), 16, 75))
    prev_offenses = int(np.random.exponential(0.6))  # most have 0, some have 1-5
    
    SUSPECT_POOL.append({
        "Suspect_ID": f"SUS-{i:04d}",
        "Suspect_Age": sus_age,
        "Suspect_Gender": sus_gender,
        "Previous_Offenses": prev_offenses,
        "Base_Crime_Category": random.choice(list(CRIME_MAPPING.keys())),
        "Associated_Gang": f"G-{random.randint(1, 15)}" if random.random() < 0.15 else None
    })

def generate_crime_dataset(num_records=10500, output_path="data/synthetic_crime_data.csv"):
    """
    Generates a realistic synthetic Karnataka State Police crime dataset
    with correlated fields, repeat offenders, and spatial clusters.
    """
    records = []
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 6, 15)
    days_range = (end_date - start_date).days
    
    # Pre-select districts based on weights
    districts = list(KARNATAKA_DISTRICTS.keys())
    district_weights = [KARNATAKA_DISTRICTS[d]["crime_weight"] for d in districts]
    chosen_districts = np.random.choice(districts, size=num_records, p=district_weights)
    
    for i in range(num_records):
        fir_id = f"FIR-{start_date.year + random.randint(0, 2)}-{10000 + i}"
        
        # District & Station selection
        district = chosen_districts[i]
        station = random.choice(KARNATAKA_DISTRICTS[district]["stations"])
        
        # Spatial placement (Gaussian offset around centroid)
        centroid = KARNATAKA_DISTRICTS[district]["coords"]
        lat = float(centroid[0] + np.random.normal(0, 0.04))
        lon = float(centroid[1] + np.random.normal(0, 0.04))
        
        # Date & Time (Seasonal patterns: property crime rises in Oct-Nov, cybercrime in 2025/2026)
        rand_days = random.randint(0, days_range)
        crime_date = start_date + timedelta(days=rand_days)
        
        # Add holiday / festival seasonal boosts
        # Dussehra/Diwali (typically Oct/Nov)
        if crime_date.month in [10, 11] and random.random() < 0.15:
            # Shift crime categories towards property crimes / theft
            crime_category = "Property Crimes"
        else:
            # Normal categorical weight
            crime_category = np.random.choice(
                list(CRIME_MAPPING.keys()),
                p=[0.15, 0.25, 0.20, 0.18, 0.10, 0.12]
            )
            
        crime_type = random.choice(CRIME_MAPPING[crime_category])
        
        # Cybercrime trend: increasing significantly in 2025 and 2026
        if crime_date.year == 2025 and random.random() < 0.1:
            crime_category = "Cyber Crimes"
            crime_type = random.choice(CRIME_MAPPING["Cyber Crimes"])
        elif crime_date.year == 2026 and random.random() < 0.2:
            crime_category = "Cyber Crimes"
            crime_type = random.choice(CRIME_MAPPING["Cyber Crimes"])
            
        crime_time = f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
        
        # Victim profile
        victim_gender = random.choice(["Male", "Female", "Other"])
        if crime_category == "Crimes Against Women":
            victim_gender = "Female"
        else:
            # 55% Male, 43% Female, 2% Other
            victim_gender = np.random.choice(["Male", "Female", "Other"], p=[0.55, 0.43, 0.02])
            
        # Age distribution
        victim_age = int(np.clip(np.random.normal(38, 14), 8, 85))
        
        # Suspect assignment (Link to suspect pool for repeat offender analysis)
        # 30% chance of a repeat offender (drawing an existing suspect with high offenses)
        if random.random() < 0.30:
            # filter suspects that align somewhat with the crime category
            sus_candidates = [s for s in SUSPECT_POOL if s["Base_Crime_Category"] == crime_category or s["Previous_Offenses"] > 1]
            if not sus_candidates:
                sus_candidates = SUSPECT_POOL
            suspect = random.choice(sus_candidates)
        else:
            suspect = random.choice(SUSPECT_POOL)
            
        sus_id = suspect["Suspect_ID"]
        sus_age = suspect["Suspect_Age"]
        sus_gender = suspect["Suspect_Gender"]
        prev_offenses = suspect["Previous_Offenses"]
        
        # Update suspect's previous offense count slightly (so the dataset is dynamically consistent)
        if random.random() < 0.1:
            suspect["Previous_Offenses"] += 1
            
        # Socio-economic index (derived from district range)
        se_range = KARNATAKA_DISTRICTS[district]["socio_economic_range"]
        socio_economic_idx = int(random.uniform(se_range[0], se_range[1]))
        
        # Investigation status
        investigation_status = np.random.choice(INVESTIGATION_STATUSES, p=STATUS_WEIGHTS)
        
        # Financial loss (correlated to crime type)
        financial_loss = 0.0
        if crime_category == "Economic Crimes":
            financial_loss = round(float(np.random.exponential(150000) + 10000), 2)
        elif crime_category == "Cyber Crimes":
            financial_loss = round(float(np.random.exponential(80000) + 5000), 2)
        elif crime_category == "Property Crimes":
            financial_loss = round(float(np.random.exponential(40000) + 1000), 2)
            
        # Crime Severity mapping
        # Cyber/Violent/Economic/Narcotics have higher severity
        if crime_category in ["Violent Crimes", "Narcotics"] or (crime_category == "Economic Crimes" and financial_loss > 100000):
            crime_severity = np.random.choice(["Medium", "High"], p=[0.3, 0.7])
        elif crime_category == "Property Crimes" or crime_category == "Crimes Against Women":
            crime_severity = np.random.choice(["Low", "Medium", "High"], p=[0.2, 0.6, 0.2])
        else:
            crime_severity = np.random.choice(["Low", "Medium"], p=[0.7, 0.3])
            
        # Risk Score Calculation (1 to 100)
        # Based on crime severity, previous offenses, socio-economic index, and financial loss
        severity_val = {"High": 40, "Medium": 25, "Low": 10}[crime_severity]
        prev_offense_val = min(prev_offenses * 8, 30)
        se_val = max(0, 100 - socio_economic_idx) * 0.2  # lower SE means higher vulnerability/risk
        loss_val = min((financial_loss / 500000) * 10, 10)
        
        risk_score = int(severity_val + prev_offense_val + se_val + loss_val)
        risk_score = min(100, max(1, risk_score))
        
        # Map risk score to Risk_Level
        if risk_score >= 65:
            risk_level = "High"
        elif risk_score >= 35:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        records.append({
            "FIR_ID": fir_id,
            "Crime_Type": crime_type,
            "Crime_Category": crime_category,
            "District": district,
            "Police_Station": station,
            "Latitude": lat,
            "Longitude": lon,
            "Date": crime_date.strftime("%Y-%m-%d"),
            "Time": crime_time,
            "Victim_Age": victim_age,
            "Victim_Gender": victim_gender,
            "Suspect_ID": sus_id,
            "Suspect_Age": sus_age,
            "Suspect_Gender": sus_gender,
            "Previous_Offenses": prev_offenses,
            "Socio_Economic_Index": socio_economic_idx,
            "Investigation_Status": investigation_status,
            "Crime_Severity": crime_severity,
            "Financial_Loss": financial_loss,
            "Risk_Level": risk_level
        })
        
    df = pd.DataFrame(records)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records in '{output_path}'.")
    return df

if __name__ == "__main__":
    generate_crime_dataset()
