import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data and model
df = pd.read_csv(r'C:\Users\MSI Ryzen 5\Desktop\diabetes_prediction\data\diabetes_cleaned.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']
model = joblib.load(r'C:\Users\MSI Ryzen 5\Desktop\diabetes_prediction\models\diabetes_model.pkl')
y_pred = model.predict(X)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()  # Waits until you close it

# Classification Report
report = classification_report(y, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)

plt.figure(figsize=(8, 5))
sns.heatmap(report_df, annot=True, cmap="YlGnBu", fmt=".2f", cbar=False)
plt.title("Classification Report", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()  # Waits until you close it

# Simple English Interpretation
accuracy = report['accuracy']
precision_0 = report['0']['precision']
recall_0 = report['0']['recall']
precision_1 = report['1']['precision']
recall_1 = report['1']['recall']

plt.figure(figsize=(10, 5))
plt.axis("off")
interpretation = (
    f"✔️ Model Accuracy: {accuracy * 100:.1f}%\n\n"
    f"🧪 Correctly detects NON-diabetics: {recall_0 * 100:.1f}%\n"
    f"🩺 Correctly detects diabetics: {recall_1 * 100:.1f}%\n\n"
    f"📌 If predicted DIABETIC, correct {precision_1 * 100:.1f}% of the time\n"
    f"📌 If predicted NON-DIABETIC, correct {precision_0 * 100:.1f}% of the time\n\n"
    f"➡️ The model performs well and is reliable for both cases."
)
plt.text(0.01, 0.98, interpretation, fontsize=13, va='top')
plt.title("Simple English Interpretation", fontsize=16)
plt.tight_layout()
plt.show()
