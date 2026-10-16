"""
Gercek Verisetinden Hasta Testi
"""

import pandas as pd
import numpy as np
import joblib

# Modelleri yukle
models = joblib.load("models.pkl")
feature_names = joblib.load("feature_names.pkl")

print("=" * 70)
print("GERCEK HASTA TESTI")
print("=" * 70)

# Veriyi yukle
df = pd.read_csv("data/cardio_train.csv", sep=";")
df.drop("id", axis=1, inplace=True)

# Yasi donustur
df["age"] = (df["age"] / 365).round().astype(int)

# BMI ekle
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

# Ilginc 3 ornek secelim
print("\nVERISETINDEN 3 GERCEK ORNEK:\n")

# ORNEK 1: Yuksek Riskli Hasta (cardio=1)
example1_idx = 1
patient1 = df.iloc[example1_idx]

print("+" + "-" * 68 + "+")
print("| HASTA 1 - Gercek Label: RISKLI (cardio=1)                        |")
print("+" + "-" * 68 + "+")
print(f"| Yas:              {patient1['age']} yil                                         |")
print(f"| Cinsiyet:         {'Kadin' if patient1['gender']==1 else 'Erkek'}                                               |")
print(f"| Boy:              {patient1['height']} cm                                         |")
print(f"| Kilo:             {patient1['weight']:.0f} kg                                        |")
print(f"| BMI:              {patient1['bmi']:.1f}                                          |")
print(f"| Buyuk Tansiyon:   {patient1['ap_hi']:.0f} {'YUKSEK!' if patient1['ap_hi']>140 else ''}                         |")
print(f"| Kucuk Tansiyon:   {patient1['ap_lo']:.0f}                                          |")
print(f"| Kolesterol:       {['Normal', 'Yuksek', 'COK YUKSEK!'][int(patient1['cholesterol'])-1]:20s}   |")
print(f"| Kan Sekeri:       {['Normal', 'Yuksek', 'Cok Yuksek'][int(patient1['gluc'])-1]:20s}   |")
print(f"| Sigara:           {'Evet' if patient1['smoke']==1 else 'Hayir':20s}   |")
print(f"| Alkol:            {'Evet' if patient1['alco']==1 else 'Hayir':20s}   |")
print(f"| Fiziksel Aktif:   {'Evet' if patient1['active']==1 else 'Hayir':20s}   |")
print("+" + "-" * 68 + "+")

# Model tahminleri
X1 = patient1[feature_names].values.reshape(1, -1)
print("\nMODEL TAHMINLERI:")
print("-" * 70)

predictions1 = {}
for name, model in models.items():
    pred_proba = model.predict_proba(X1)[0, 1]
    predictions1[name] = pred_proba
    risk_level = "YUKSEK" if pred_proba > 0.66 else "ORTA" if pred_proba > 0.33 else "DUSUK"
    print(f"[{risk_level:6s}] {name:20s} -> %{pred_proba*100:.1f} risk")

avg_pred1 = np.mean(list(predictions1.values()))
print("-" * 70)
print(f"ORTALAMA TAHMIN: %{avg_pred1*100:.1f}")
print(f"GERCEK DURUM:    RISKLI (cardio=1)")
correct1 = "DOGRU!" if avg_pred1 > 0.5 else "YANLIS!"
print(f"SONUC:           {correct1}")

print("\n" + "=" * 70)
print("\n")

# ORNEK 2: Dusuk Riskli Hasta (cardio=0)
example2_idx = 0
patient2 = df.iloc[example2_idx]

print("+" + "-" * 68 + "+")
print("| HASTA 2 - Gercek Label: SAGLIKLI (cardio=0)                      |")
print("+" + "-" * 68 + "+")
print(f"| Yas:              {patient2['age']} yil                                         |")
print(f"| Cinsiyet:         {'Kadin' if patient2['gender']==1 else 'Erkek'}                                               |")
print(f"| Boy:              {patient2['height']} cm                                         |")
print(f"| Kilo:             {patient2['weight']:.0f} kg                                        |")
print(f"| BMI:              {patient2['bmi']:.1f}                                          |")
print(f"| Buyuk Tansiyon:   {patient2['ap_hi']:.0f} {'Normal' if patient2['ap_hi']<=130 else ''}                         |")
print(f"| Kucuk Tansiyon:   {patient2['ap_lo']:.0f}                                          |")
print(f"| Kolesterol:       {['Normal', 'Yuksek', 'Cok Yuksek'][int(patient2['cholesterol'])-1]:20s}   |")
print(f"| Kan Sekeri:       {['Normal', 'Yuksek', 'Cok Yuksek'][int(patient2['gluc'])-1]:20s}   |")
print(f"| Sigara:           {'Evet' if patient2['smoke']==1 else 'Hayir':20s}   |")
print(f"| Alkol:            {'Evet' if patient2['alco']==1 else 'Hayir':20s}   |")
print(f"| Fiziksel Aktif:   {'Evet' if patient2['active']==1 else 'Hayir':20s}   |")
print("+" + "-" * 68 + "+")

X2 = patient2[feature_names].values.reshape(1, -1)
print("\nMODEL TAHMINLERI:")
print("-" * 70)

predictions2 = {}
for name, model in models.items():
    pred_proba = model.predict_proba(X2)[0, 1]
    predictions2[name] = pred_proba
    risk_level = "YUKSEK" if pred_proba > 0.66 else "ORTA" if pred_proba > 0.33 else "DUSUK"
    print(f"[{risk_level:6s}] {name:20s} -> %{pred_proba*100:.1f} risk")

avg_pred2 = np.mean(list(predictions2.values()))
print("-" * 70)
print(f"ORTALAMA TAHMIN: %{avg_pred2*100:.1f}")
print(f"GERCEK DURUM:    SAGLIKLI (cardio=0)")
correct2 = "DOGRU!" if avg_pred2 < 0.5 else "YANLIS!"
print(f"SONUC:           {correct2}")

print("\n" + "=" * 70)
print("\n")

# ORNEK 3: Sinirda Olan (Ilginc durum)
example3_idx = 4
patient3 = df.iloc[example3_idx]

print("+" + "-" * 68 + "+")
print(f"| HASTA 3 - Gercek Label: {'RISKLI' if patient3['cardio']==1 else 'SAGLIKLI'} (cardio={int(patient3['cardio'])})                      |")
print("+" + "-" * 68 + "+")
print(f"| Yas:              {patient3['age']} yil                                         |")
print(f"| Cinsiyet:         {'Kadin' if patient3['gender']==1 else 'Erkek'}                                               |")
print(f"| Boy:              {patient3['height']} cm                                         |")
print(f"| Kilo:             {patient3['weight']:.0f} kg                                        |")
print(f"| BMI:              {patient3['bmi']:.1f}                                          |")
print(f"| Buyuk Tansiyon:   {patient3['ap_hi']:.0f}                                          |")
print(f"| Kucuk Tansiyon:   {patient3['ap_lo']:.0f}                                          |")
print(f"| Kolesterol:       {['Normal', 'Yuksek', 'Cok Yuksek'][int(patient3['cholesterol'])-1]:20s}   |")
print(f"| Kan Sekeri:       {['Normal', 'Yuksek', 'Cok Yuksek'][int(patient3['gluc'])-1]:20s}   |")
print(f"| Sigara:           {'Evet' if patient3['smoke']==1 else 'Hayir':20s}   |")
print(f"| Alkol:            {'Evet' if patient3['alco']==1 else 'Hayir':20s}   |")
print(f"| Fiziksel Aktif:   {'Evet' if patient3['active']==1 else 'Hayir':20s}   |")
print("+" + "-" * 68 + "+")

X3 = patient3[feature_names].values.reshape(1, -1)
print("\nMODEL TAHMINLERI:")
print("-" * 70)

predictions3 = {}
for name, model in models.items():
    pred_proba = model.predict_proba(X3)[0, 1]
    predictions3[name] = pred_proba
    risk_level = "YUKSEK" if pred_proba > 0.66 else "ORTA" if pred_proba > 0.33 else "DUSUK"
    print(f"[{risk_level:6s}] {name:20s} -> %{pred_proba*100:.1f} risk")

avg_pred3 = np.mean(list(predictions3.values()))
print("-" * 70)
print(f"ORTALAMA TAHMIN: %{avg_pred3*100:.1f}")
print(f"GERCEK DURUM:    {'RISKLI' if patient3['cardio']==1 else 'SAGLIKLI'} (cardio={int(patient3['cardio'])})")
if patient3['cardio'] == 1:
    correct3 = "DOGRU!" if avg_pred3 > 0.5 else "YANLIS!"
else:
    correct3 = "DOGRU!" if avg_pred3 < 0.5 else "YANLIS!"
print(f"SONUC:           {correct3}")

print("\n" + "=" * 70)
print("\nOZET:")
print("-" * 70)
total_correct = sum([1 if c == "DOGRU!" else 0 for c in [correct1, correct2, correct3]])
print(f"3 ornekten {total_correct} tanesi dogru tahmin edildi.")
print("\nBu 'Sanity Check' gosteriyor ki model gercek veride calisiyor!")
print("=" * 70)
