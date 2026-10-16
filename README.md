# Kalp Krizi Risk Tahmini ve Açıklanabilir Yapay Zeka

Kardiyovasküler hastalık riskini tahmin eden bir makine öğrenmesi uygulaması. Dört modelin ağırlıklı ortalamasını kullanır ve sonucu SHAP ile açıklar.

Bu uygulama tıbbi tanı, tedavi veya acil karar aracı değildir. Gösterilen risk yüzdesi ve öneriler eğitim amaçlıdır; bir hekim değerlendirmesinin yerini tutmaz.

## Kurulum

Python 3.8 veya üzeri gerekir.

```bash
cd cardiovascular-disease_-project
pip install -r requirements.txt
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` açılır.

## Kullanım

1. Yaş, cinsiyet, boy, kilo, tansiyon, kolesterol, kan şekeri, sigara, alkol ve fiziksel aktivite bilgilerini girin.
2. Tahmin butonuna basın.
3. Risk yüzdesini, SHAP grafiğini ve kişiselleştirilmiş önerileri inceleyin.

## Proje yapısı

```
cardiovascular-disease_-project/
├── app.py                  # Streamlit arayüzü
├── final_update_models.py  # Modelleri yeniden eğitir
├── test_real_patient.py    # Veri setinden örnek hasta testi
├── models.pkl              # Eğitilmiş 4 model
├── explainers.pkl          # SHAP explainer nesneleri
├── feature_names.pkl       # Özellik adları
├── data/
│   └── cardio_train.csv    # Eğitim verisi (70.000 kayıt)
├── notebooks/
│   └── analysis.ipynb      # Veri analizi
├── requirements.txt
└── README.md
```

Uygulama açılışta üç `.pkl` dosyasını yükler. Yeniden eğitim ve `test_real_patient.py` için `data/cardio_train.csv` gerekir.

## Veri seti

`data/cardio_train.csv`, Svetlana Ulianova tarafından paylaşılan Cardiovascular Disease veri setidir.

- Kaynak: [Kaggle — Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- Kayıt: 70.000 satır, 11 özellik + hedef (`cardio`)
- Ayırıcı: noktalı virgül (`;`). `age` gün cinsindendir; eğitim sırasında yıla çevrilir.
- Kaggle üzerindeki lisans: **Unknown**

Dosya, modellerin yeniden eğitilebilmesi ve test betiğinin çalışması için bu repoda durur. Veriyi başka bir yerde kullanırken Kaggle sayfasındaki güncel şartlara bakın.

Özellikler: yaş, boy, kilo, cinsiyet, büyük tansiyon (`ap_hi`), küçük tansiyon (`ap_lo`), kolesterol, kan şekeri, sigara, alkol, fiziksel aktivite. Hedef, muayene anında kardiyovasküler hastalık olup olmadığıdır (`cardio`).

## Modeller

| Model | ROC-AUC | Ensemble ağırlığı |
|-------|---------|-------------------|
| XGBoost | 0.804 | %35 |
| Gradient Boosting | 0.801 | %30 |
| LightGBM | 0.798 | %25 |
| AdaBoost | 0.789 | %10 |

Açıklama için XGBoost, Gradient Boosting ve LightGBM üzerinde SHAP `TreeExplainer`, AdaBoost üzerinde `KernelExplainer` kullanılır.

## Teknolojiler

| Kategori | Teknoloji |
|----------|-----------|
| Modeller | XGBoost, LightGBM, Gradient Boosting, AdaBoost |
| Açıklanabilirlik | SHAP |
| Arayüz | Streamlit |
| Veri | Pandas, NumPy, Scikit-learn |
| Grafik | Plotly, Matplotlib, Seaborn |

## Sorun giderme

Eksik paket:

```bash
pip install -r requirements.txt
```

8501 portu doluysa:

```bash
streamlit run app.py --server.port 8502
```

Model dosyaları yoksa (eğitim birkaç dakika sürebilir):

```bash
python final_update_models.py
```
