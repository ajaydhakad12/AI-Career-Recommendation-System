import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Step 1: Load dataset
df = pd.read_csv("dataset9000.csv")

# Step 2: Separate features and target
X = df.drop("Role", axis=1)   # Features (skills)
y = df["Role"]                # Target (career)

# Step 3: Encode categorical feature columns (skills)
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Step 4: Encode target labels (career names)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Step 6: Train RandomForest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate model
accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)

# Step 8: Save model and encoder
joblib.dump(model, "career_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("Training complete. Model and encoder saved.")
