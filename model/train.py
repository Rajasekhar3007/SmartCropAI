import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv("../dataset/crop_data.csv")

# Clean labels
df['label'] = df['label'].str.strip()

# Features and target
X = df.drop("label", axis=1)
y = df["label"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Model
model = XGBClassifier()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "../saved_models/crop_model.pkl")
joblib.dump(le, "../saved_models/label_encoder.pkl")

print("Model saved!")
