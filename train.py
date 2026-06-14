import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("data/weather_data.csv")

print("Dataset Shape:", df.shape)

# ==========================
# Remove Timestamp
# ==========================
df = df.drop("Timestamp", axis=1)

# ==========================
# Features and Target
# ==========================
X = df.drop("Weather_Label", axis=1)

y = df["Weather_Label"]

# ==========================
# Encode Labels
# ==========================
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nWeather Classes:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{i} -> {label}")

# ==========================
# Train/Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ==========================
# Train Model
# ==========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# Evaluate Model
# ==========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# ==========================
# Save Model
# ==========================
joblib.dump(model, "model/weather_model.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")

print("\nModel Saved Successfully!")