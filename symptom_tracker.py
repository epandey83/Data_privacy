import pandas as pd
import random
from datetime import datetime, timedelta, date

num_users = 100

start_year = 2018
end_year = 2023

def random_date(start_year, end_year):
    start = datetime(start_year, 1, 1, 0, 0, 0)
    end = datetime(end_year, 12, 31, 23, 59, 59)
    delta = end - start
    random_second = random.randint(0, delta.total_seconds())
    return start + timedelta(seconds=random_second)



# List of symptoms to choose from
symptoms = ["Depression", "Stress", "Panic Attacks"]

# List of notes to choose from
notes = ["Feeling overwhelmed", "Difficulty concentrating", "Trouble sleeping", "Lack of appetite",
         "Feeling hopeless", "Lack of energy", "Increased heart rate", "Sweating", "Shaking"]

# Generate synthetic data for 100 users
data = []
for i in range(1, num_users+1):
    user_id = i
    symptom = random.choice(symptoms)
    severity = random.randint(1, 63)
    frequency = random.choice(["Daily", "2-3 times per week", "Once a week", "Once every two weeks", "Monthly"])
    note = random.choice(notes)
    timestamp = random_date(start_year, end_year)
    #data.append([user_id, name, email, age, gender, payment_info, symptom, severity, frequency, note, timestamp])
    data.append([user_id, symptom, severity, frequency, note, timestamp])
0
# Create a Pandas DataFrame from the data
#df = pd.DataFrame(data, columns=["User ID", "Name", "email", "Age", "gender","payment method", "Symptom", "Severity (1-10)", "Frequency", "Notes", "Timestamp"])
df = pd.DataFrame(data, columns=["User ID", "Symptom", "Severity (BDI score)", "Frequency", "Notes", "Timestamp"])

# Store the DataFrame in a CSV file
df.to_csv("symptom_data.csv", index=False)



# Convert the frequency values to categories
severity_mapping = {
    (1,10): 'These ups and downs are considered normal',
    (11,16): 'Mild mood disturbance',
    (17,20): 'Borderline clinical depression',
    (21,30): 'Moderate depression',
    (31,40): 'Severe depression',
    (41,63): 'Extreme depression'
    
}

# Convert the frequency values to categories
frequency_mapping = {
    1: 'Daily',
    2: 'Weekly',
    3: 'Monthly',
    4: 'Occasionally'
}
#df['Frequency'] = df['Frequency'].map(frequency_mapping)
#df['Severity (BDI score)'] = df['Severity (BDI score)'].map(severity_mapping)
df['Severity (BDI score)'] = df['Severity (BDI score)'].apply(lambda x: next((v for k, v in severity_mapping.items() if x in range(k[0], k[1]+1)), None))
df = df.rename(columns={'Severity (BDI score)': 'Severity'})


# Round the timestamp values to the nearest month
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Timestamp'] = df['Timestamp'].dt.to_period('M').dt.to_timestamp()
df['Month'] = df['Timestamp'].dt.month

# Map the timestamp values to the season of the year
season_mapping = {
    1: 'Winter',
    2: 'Winter',
    3: 'Spring',
    4: 'Spring',
    5: 'Spring',
    6: 'Summer',
    7: 'Summer',
    8: 'Summer',
    9: 'Fall',
    10: 'Fall',
    11: 'Fall',
    12: 'Winter'
}
df['Season'] = df['Month'].map(season_mapping)
df['Timestamp_year_season'] =  df['Month'].map(season_mapping) + ' ' + df['Timestamp'].dt.year.astype(str)
df.drop(['Month'], axis=1, inplace=True)
df.drop(['Season'], axis=1, inplace=True)
df.drop(['Timestamp'], axis=1, inplace=True)
df = df.rename(columns={'Timestamp_year_season': 'Timestamp'})



# Export the de-identified data to a new CSV file
df.to_csv('deidentified_symptom_data.csv', index=False)

