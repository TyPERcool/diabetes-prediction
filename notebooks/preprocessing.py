import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ensure the data directory exists
data_dir = Path(__file__).resolve().parent.parent / 'data'
data_dir.mkdir(parents=True, exist_ok=True)  # Creates the directory if it doesn't exist

# Load dataset
df = pd.read_csv(data_dir / 'diabetes.csv')

# Replace 0s in certain columns with NaN
columns_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_to_clean] = df[columns_to_clean].replace(0, pd.NA)

# Fill with median
df.fillna(df.median(), inplace=True)

# Optional scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('Outcome', axis=1))
X_scaled = pd.DataFrame(X_scaled, columns=df.columns[:-1])

X_scaled['Outcome'] = df['Outcome'].values

# Save cleaned data
X_scaled.to_csv(data_dir / 'diabetes_cleaned.csv', index=False)

print("✅ Preprocessed data saved to diabetes_cleaned.csv")
