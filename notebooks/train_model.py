import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv(r'C:\Users\MSI Ryzen 5\Desktop\diabetes_prediction\data\diabetes_cleaned.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, r'C:\Users\MSI Ryzen 5\Desktop\diabetes_prediction\models\diabetes_model.pkl')
print("✅ Model saved to models/diabetes_model.pkl")
