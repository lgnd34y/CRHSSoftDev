import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df_main = pd.read_csv("asl_data.csv", header=None)


df_feedback = pd.read_csv("feedback_data.csv", header=None)
df_main = pd.concat([df_main, df_feedback, df_feedback, df_feedback], ignore_index=True)
print("Total training samples:", len(df_main))


x = df_main.iloc[:, 1:]
y = df_main.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

model = RandomForestClassifier(n_estimators=600,max_depth=30,min_samples_leaf=2,random_state=42,n_jobs=-1)

model.fit(x_train, y_train)


print(f"Accuracy: {model.score(x_train, y_train)}")
joblib.dump(model, "asl_model.pkl")
