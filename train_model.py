import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ---------------- LOAD BASE DATA ----------------
df_main = pd.read_csv("asl_data.csv", header=None)

# ---------------- LOAD FEEDBACK DATA ----------------
if os.path.exists("feedback_data.csv"):
    df_feedback = pd.read_csv("feedback_data.csv", header=None)
    print(f"Loaded {len(df_feedback)} feedback samples")

    # Weight feedback samples higher (VERY IMPORTANT)
    df_feedback = pd.concat([df_feedback] * 3, ignore_index=True)

    df = pd.concat([df_main, df_feedback], ignore_index=True)
else:
    df = df_main

print("Total training samples:", len(df))

# ---------------- FEATURES / LABELS ----------------
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- MODEL ----------------
model = RandomForestClassifier(
    n_estimators=600,
    max_depth=30,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# ---------------- TRAIN ----------------
print("Training model...")
model.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------- SAVE ----------------
joblib.dump(model, "asl_model.pkl")
print("\nModel saved as asl_model.pkl")
