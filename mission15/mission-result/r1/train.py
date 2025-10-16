# -*- coding: utf-8 -*-
import os, json, shutil, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

DATA_DIR = "/data"
ARTIFACTS_DIR = "/artifacts"

TRAIN_PATH = os.path.join(DATA_DIR, "mission15_train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "mission15_test.csv")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "train_report.json")
LOG_PATH    = os.path.join(ARTIFACTS_DIR, "train_log.txt")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

df = pd.read_csv(TRAIN_PATH)

target_col = "Performance Index"
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].copy()
y = df[target_col].copy()

numeric_features = ["Hours Studied","Previous Scores","Sleep Hours","Sample Question Papers Practiced"]
categorical_features = ["Extracurricular Activities"]

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

reg_pipeline = Pipeline(steps=[
    ("prep", preprocessor),
    ("rf", model)
])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg_pipeline.fit(X_train, y_train)

valid_pred = reg_pipeline.predict(X_valid)
rmse = mean_squared_error(y_valid, valid_pred, squared=False)

joblib.dump(reg_pipeline, MODEL_PATH)

report = {
    "rmse_valid": float(rmse),
    "n_train": int(len(X_train)),
    "n_valid": int(len(X_valid)),
    "model_type": "RandomForestRegressor",
    "sklearn_version": "1.4.2",
    "notes": "Preprocessor(ColumnTransformer)+RandomForest pipeline"
}
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write(f"[INFO] Validation RMSE: {rmse:.4f}\n")

if os.path.exists(TEST_PATH):
    shutil.copy2(TEST_PATH, os.path.join(ARTIFACTS_DIR, "mission15_test.csv"))

print("[DONE] model.pkl 생성 및 보고서 저장 완료")
print(f"RMSE(valid) = {rmse:.4f}")
print(f"Saved: {MODEL_PATH}")
