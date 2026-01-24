from sklearnex import patch_sklearn
patch_sklearn()
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import config

df_main = pd.read_csv(config.data, header=None)
df_feedback = pd.read_csv(config.feedback_data, header=None)

df_main = pd.concat([df_main, df_feedback, df_feedback, df_feedback], ignore_index=True)

df_words = df_main[df_main[0].isin(config.word_labels)]

df_main = pd.concat([df_main, df_words, df_words, df_words, df_words, df_words], ignore_index=True)

print(f"Total training samples: {len(df_main)}")

x = df_main.iloc[:, 1:]
y = df_main.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, max_depth=40, random_state=42, n_jobs=-1,class_weight="balanced")

model.fit(x_train, y_train)

print(f"Test Accuracy: {model.score(x_test, y_test):.4f}")
joblib.dump(model, "asl_model.pkl")
