import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

mlflow.autolog()

print("Memuat data dari Link Permanent (PASTI JALAN)...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'HeartDisease']

df = pd.read_csv(url, names=column_names, na_values="?")

df = df.dropna()

df['HeartDisease'] = df['HeartDisease'].apply(lambda x: 1 if x > 0 else 0)

X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Mulai Training...")
with mlflow.start_run(run_name="Heart_Disease_Final_Model") as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Akurasi: {acc}")

    with open("last_run_id.txt", "w") as f:
        f.write(run.info.run_id)

print("✅ Model SELESAI. Run ID Aman!")