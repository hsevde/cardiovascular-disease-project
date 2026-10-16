#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model Güncelleme - AdaBoost için özel SHAP çözümü"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("Veri yukleniyor...", flush=True)

# Veri hazırlama
df = pd.read_csv("data/cardio_train.csv", sep=";")
df.drop("id", axis=1, inplace=True)
df["age"] = (df["age"] / 365).round().astype(int)
df = df[df["ap_lo"] < df["ap_hi"]]
df = df[(df["ap_hi"] > 50) & (df["ap_hi"] < 250)]
df = df[(df["ap_lo"] > 30) & (df["ap_lo"] < 200)]
df = df[(df["height"] > 120) & (df["height"] < 220)]
df = df[(df["weight"] > 30) & (df["weight"] < 200)]
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

X = df.drop("cardio", axis=1)
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Veri hazir: {df.shape}", flush=True)
print("Modeller egitiliyor...", flush=True)

# Modeller
models = {
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42, verbose=0),
    "LightGBM": LGBMClassifier(n_estimators=300, max_depth=6, random_state=42, verbosity=-1),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42)
}

for name, clf in models.items():
    print(f"  {name}...", flush=True)
    clf.fit(X_train, y_train)

print("\nSHAP explainers olusturuluyor...", flush=True)

explainer_dict = {}

# AdaBoost için özel çözüm: sample data ile KernelExplainer
sample_data = shap.sample(X_train, 100)

for name, clf in models.items():
    print(f"  {name} explainer...", flush=True)
    try:
        if name == "AdaBoost":
            # AdaBoost için KernelExplainer
            explainer_dict[name] = shap.KernelExplainer(clf.predict_proba, sample_data)
        else:
            # Diğerleri için TreeExplainer
            explainer_dict[name] = shap.TreeExplainer(clf)
        print(f"  {name} OK!", flush=True)
    except Exception as e:
        print(f"  {name} explainer hatasi (atlaniyor): {e}", flush=True)

print("\nKaydediliyor...", flush=True)

joblib.dump(models, "models.pkl")
joblib.dump(explainer_dict, "explainers.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("\n=== TAMAMLANDI ===", flush=True)
print(f"Toplam {len(models)} model kaydedildi:", list(models.keys()), flush=True)
print("\nStreamlit uygulamasini yeniden baslatin!", flush=True)

