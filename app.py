import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO



models = joblib.load("models.pkl")
explainers = joblib.load("explainers.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(
    page_title="Kalp Krizi Risk Tahmini",
    page_icon="❤️",
    layout="wide"
)


st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .risk-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeaa7;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .suggestion-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff4e5;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="main-header">
    <h1>❤️ Kalp Krizi Risk Tahmini ve Model Karşılaştırması</h1>

</div>
""", unsafe_allow_html=True)


pretty_names = {
    "age": "Yaş",
    "ap_hi": "Büyük tansiyon",
    "ap_lo": "Küçük tansiyon",
    "cholesterol": "Kolesterol seviyesi",
    "gluc": "Kan şekeri seviyesi",
    "bmi": "Vücut kitle indeksi (BMI)",
    "weight": "Kilo",
    "height": "Boy",
    "gender": "Cinsiyet",
    "smoke": "Sigara kullanımı",
    "alco": "Alkol kullanımı",
    "active": "Fiziksel aktivite"
}


st.subheader(" Hasta Bilgilerini Girin")

col1, col2, col3 = st.columns(3)

input_values = {}

with col1:
    st.markdown("#####  Demografik Bilgiler")
    if "age" in feature_names:
        age_years = st.slider("Yaş (yıl)", 20, 90, 50)
        input_values["age"] = age_years
    
    if "gender" in feature_names:
        gender = st.radio(
            "Cinsiyet",
            [1, 2],
            format_func=lambda x: "Kadın" if x == 1 else "Erkek"
        )
        input_values["gender"] = gender
    
    if "height" in feature_names:
        height_val = st.slider("Boy (cm)", 140, 210, 170)
        input_values["height"] = height_val
    
    if "weight" in feature_names:
        weight_val = st.slider("Kilo (kg)", 40, 150, 75)
        input_values["weight"] = weight_val
    
    if "bmi" in feature_names:
        if height_val is not None and weight_val is not None:
            bmi = weight_val / ((height_val / 100) ** 2)
        else:
            bmi = st.number_input("Vücut Kitle İndeksi (BMI)", min_value=10.0, max_value=60.0, value=25.0)
        input_values["bmi"] = bmi
        st.info(f" BMI: {bmi:.1f}")

with col2:
    st.markdown("##### 🩺 Klinik Ölçümler")
    if "ap_hi" in feature_names:
        ap_hi = st.slider("Büyük Tansiyon (ap_hi)", 80, 220, 130)
        input_values["ap_hi"] = ap_hi
        if ap_hi > 180:
            st.error("⚠️ Çok yüksek!")
        elif ap_hi > 140:
            st.warning("⚠️ Yüksek")
    
    if "ap_lo" in feature_names:
        ap_lo = st.slider("Küçük Tansiyon (ap_lo)", 50, 140, 80)
        input_values["ap_lo"] = ap_lo
    
    if "cholesterol" in feature_names:
        chol_val = st.selectbox(
            "Kolesterol Seviyesi",
            [1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "Yüksek", 3: "Çok yüksek"}[x]
        )
        input_values["cholesterol"] = chol_val
    
    if "gluc" in feature_names:
        gluc_val = st.selectbox(
            "Kan Şekeri Seviyesi",
            [1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "Yüksek", 3: "Çok yüksek"}[x]
        )
        input_values["gluc"] = gluc_val

with col3:
    st.markdown("#####  Yaşam Tarzı")
    if "smoke" in feature_names:
        smoke = st.selectbox(
            "Sigara İçiyor musunuz?",
            [0, 1],
            format_func=lambda x: "Hayır" if x == 0 else "Evet"
        )
        input_values["smoke"] = smoke
    
    if "alco" in feature_names:
        alco = st.selectbox(
            "Alkol Kullanımı",
            [0, 1],
            format_func=lambda x: "Hayır" if x == 0 else "Evet"
        )
        input_values["alco"] = alco
    
    if "active" in feature_names:
        active = st.selectbox(
            "Fiziksel olarak aktif misiniz?",
            [0, 1],
            format_func=lambda x: "Evet" if x == 1 else "Hayır"
        )
        input_values["active"] = active

st.markdown("---")


row_data = []
for feat in feature_names:
    if feat in input_values:
        row_data.append(input_values[feat])
    else:
        row_data.append(0)

row = pd.DataFrame([row_data], columns=feature_names)


st.markdown("<center>", unsafe_allow_html=True)
calculate_button = st.button("✅ Riski Hesapla ve Analiz Et", type="primary", use_container_width=True)
st.markdown("</center>", unsafe_allow_html=True)

if calculate_button:
    with st.spinner('🔄 Modeller çalışıyor ve analiz ediliyor...'):
        model_results = []
        shap_explanation = None
        main_model_name = None

        for name, model in models.items():
            proba = model.predict_proba(row)[0, 1]
            explainer = explainers.get(name, None)
            top_features = []
            shap_values_raw = None

            if explainer is not None:
                shap_values = explainer(row)
                values = shap_values.values[0]
                
                if values.ndim == 2:
                    values = values[:, 1]
                
                # SHAP için sakla (ana model için)
                if name.lower().startswith("xgboost"):
                    shap_explanation = shap_values
                    shap_values_raw = values
                    main_model_name = name

                idx_sorted = np.argsort(np.abs(values))[::-1][:3]

                for i, idx in enumerate(idx_sorted):
                    idx = int(idx)
                    if idx >= len(feature_names) or idx >= len(values):
                        continue
                    feat_name = feature_names[idx]
                    shap_val = float(values[idx])
                    feat_input = input_values.get(feat_name, None)
                    top_features.append((feat_name, shap_val, feat_input))

            model_results.append({
                "Model": name,
                "Risk Olasılığı": float(proba),
                "Top Özellikler": top_features,
                "SHAP Values": shap_values_raw
            })

        # WEIGHTED ENSEMBLE (Test performansına göre ağırlıklar)
        weights = {
            "XGBoost": 0.35,           # ROC-AUC: 0.804 (En iyi)
            "Gradient Boosting": 0.30,  # ROC-AUC: 0.801
            "LightGBM": 0.25,          # ROC-AUC: 0.798
            "AdaBoost": 0.10           # ROC-AUC: 0.789 (En zayıf)
        }
        
        # Ağırlıklı ortalama hesapla
        ensemble_risk = sum(
            r["Risk Olasılığı"] * weights.get(r["Model"], 0.25)
            for r in model_results
        )
        
        # Ana sonuç: Ensemble
        risk = ensemble_risk
        main_model_name = "Ensemble (Weighted Average)"
        
        # XGBoost'u referans için sakla (SHAP için)
        main_result = None
        for r in model_results:
            if r["Model"].lower().startswith("xgboost"):
                main_result = r
                break
        
        if main_result is None:
            main_result = max(model_results, key=lambda x: x["Risk Olasılığı"])
        if risk < 0.33:
            level = "DÜŞÜK"
            color_emoji = "🟢"
            risk_class = "risk-low"
        elif risk < 0.66:
            level = "ORTA"
            color_emoji = "🟡"
            risk_class = "risk-medium"
        else:
            level = "YÜKSEK"
            color_emoji = "🔴"
            risk_class = "risk-high"

    
    tab1, tab2, tab3 = st.tabs([" Risk Analizi", " SHAP Açıklaması", " Kişiselleştirilmiş Öneriler"])

    with tab1:
        st.subheader("📊 Risk Değerlendirmesi")
        
        # Ana Risk Kartı
        st.markdown(f"""
        <div class="risk-card {risk_class}">
            {color_emoji} KALP KRİZİ RİSKİ: %{risk*100:.1f} - {level} RİSK
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Tahmin Yöntemi:** `{main_model_name}`")
        st.info("💡 Risk tahmini, 4 boosting modelinin ağırlıklı ortalamasıdır (XGBoost: %35, GB: %30, LightGBM: %25, AdaBoost: %10)")
        
        # Progress bar
        st.markdown("##### Risk Seviye Göstergesi")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Yüzdesi"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if risk > 0.66 else "orange" if risk > 0.33 else "green"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "lightyellow"},
                    {'range': [66, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("---")
        
        # Model karşılaştırma
        st.subheader(" Model Karşılaştırması")
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            # Tablo
            table_data = {
                "Model": [r["Model"] for r in model_results],
                "Risk (%)": [f"%{round(r['Risk Olasılığı'] * 100, 1)}" for r in model_results],
                "Risk (0-1)": [round(r["Risk Olasılığı"], 3) for r in model_results],
            }
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        
        with col_right:
            # Bar chart
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=[r["Model"] for r in model_results],
                y=[r["Risk Olasılığı"] * 100 for r in model_results],
                marker_color=['#667eea' if r["Model"] == main_model_name else '#95a5a6' for r in model_results],
                text=[f"%{r['Risk Olasılığı']*100:.1f}" for r in model_results],
                textposition='auto',
            ))
            fig_bar.update_layout(
                title="Modellerin Risk Tahminleri",
                yaxis_title="Risk (%)",
                xaxis_title="Model",
                height=300
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.subheader(" Açıklanabilir Yapay Zeka (SHAP Analizi)")
        
        st.markdown("""
        SHAP (SHapley Additive exPlanations) değerleri, her özelliğin model kararına olan katkısını gösterir.
        - **Pozitif değerler (kırmızı)**: Riski artıran faktörler
        - **Negatif değerler (mavi)**: Riski azaltan faktörler
        """)
        
        if shap_explanation is not None:
            # SHAP Waterfall Plot
            st.markdown("#####  SHAP Waterfall Plot")
            st.markdown("Bu grafik, her özelliğin tahmin üzerindeki etkisini adım adım gösterir:")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_explanation[0], max_display=10, show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close()
            
            st.markdown("---")
            
            # En etkili özellikler
            st.markdown("#####  En Etkili Risk Faktörleri")
            
            if main_result["Top Özellikler"]:
                for i, (feat_name, shap_val, feat_input) in enumerate(main_result["Top Özellikler"], 1):
                    direction = "artırmıştır ⬆️" if shap_val > 0 else "azaltmıştır ⬇️"
                    feat_label = pretty_names.get(feat_name, feat_name)
                    
                    if feat_input is None:
                        value_text = "girilen değer"
                    else:
                        value_text = str(feat_input)
                    
                    emoji = "🔴" if shap_val > 0 else "🟢"
                    
                    st.markdown(f"""
                    <div class="metric-box">
                        <strong>{emoji} #{i}: {feat_label}</strong><br>
                        Değer: <strong>{value_text}</strong><br>
                        Etki: Riski <strong>{direction}</strong> (SHAP: {shap_val:.3f})
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("SHAP açıklaması bulunamadı.")

    with tab3:
        st.subheader("💡 Kişiselleştirilmiş Sağlık Önerileri")
        
        # Akıllı öneriler sistemi
        suggestions = []
        warnings = []
        
        # Tansiyon kontrolü
        if "ap_hi" in input_values:
            ap_hi_val = input_values["ap_hi"]
            if ap_hi_val > 180:
                warnings.append("⚠️ **ACİL:** Büyük tansiyonunuz çok yüksek (>180)! Derhal bir sağlık kuruluşuna başvurun!")
                suggestions.append("🏥 Hipertansiyon için düzenli ilaç kullanımı ve hekim takibi şarttır.")
            elif ap_hi_val > 140:
                warnings.append("⚠️ Büyük tansiyonunuz yüksek. Hekim kontrolü önerilir.")
                suggestions.append("🧂 Tuz tüketiminizi azaltın (günde 5g'dan az).")
                suggestions.append("🏃 Düzenli egzersiz yapın (haftada 150 dakika orta tempolu).")
            elif ap_hi_val <= 120:
                suggestions.append("✅ Tansiyonunuz ideal seviyede, böyle devam edin!")
        
        # BMI kontrolü
        if "bmi" in input_values:
            bmi_val = input_values["bmi"]
            if bmi_val > 30:
                warnings.append("⚠️ BMI'niz obezite seviyesinde (>30).")
                suggestions.append("🥗 Dengeli beslenme programı ve diyet desteği alın.")
                suggestions.append("🏃 Fiziksel aktiviteyi artırın, haftada 5 gün 30 dakika yürüyüş yapın.")
            elif bmi_val > 25:
                suggestions.append("📊 BMI'niz hafif kilolu kategorisinde. Kilo kontrolüne dikkat edin.")
            elif 18.5 <= bmi_val <= 25:
                suggestions.append("✅ BMI'niz sağlıklı aralıkta!")
        
        # Kolesterol
        if "cholesterol" in input_values:
            chol_val = input_values["cholesterol"]
            if chol_val == 3:
                warnings.append("⚠️ Kolesterolünüz çok yüksek.")
                suggestions.append("🥑 Omega-3 içeren gıdalar tüketin (balık, ceviz).")
                suggestions.append("🚫 Doymuş yağlardan kaçının (kızartma, fast food).")
            elif chol_val == 2:
                suggestions.append("📊 Kolesterolünüz yüksek. Beslenmede doymuş yağları azaltın.")
        
        # Kan şekeri
        if "gluc" in input_values:
            gluc_val = input_values["gluc"]
            if gluc_val >= 2:
                warnings.append("⚠️ Kan şekeriniz yüksek. Diyabet kontrolü yaptırın.")
                suggestions.append("🍏 Şeker ve rafine karbonhidrat tüketimini azaltın.")
                suggestions.append("🥦 Lifli gıdalar ve sebze ağırlıklı beslenin.")
        
        # Sigara
        if "smoke" in input_values and input_values["smoke"] == 1:
            warnings.append("⚠️ Sigara kalp krizi riskini 2-4 kat artırır!")
            suggestions.append("🚭 Sigara bırakma programlarına katılın. İlk 1 yıl içinde risk %50 azalır!")
        
        # Fiziksel aktivite
        if "active" in input_values and input_values["active"] == 0:
            suggestions.append("🏃 Fiziksel aktiviteyi artırın! Haftada en az 150 dakika orta tempolu egzersiz yapın.")
            suggestions.append("🚶 Günlük 10,000 adım hedefleyin.")
        
        # Alkol
        if "alco" in input_values and input_values["alco"] == 1:
            suggestions.append("🍷 Alkol tüketimini sınırlandırın (günde max 1-2 standart içki).")
        
        # Yaş
        if "age" in input_values and input_values["age"] > 55:
            suggestions.append("👴 Yaşınız nedeniyle düzenli kardiyoloji kontrolü önerilir (yılda 1 kez).")
        
        # Genel öneriler
        if risk > 0.66:
            warnings.append("🔴 **YÜKSEK RİSK**: Mutlaka bir kardiyolog ile görüşün!")
            suggestions.append("💊 İlaç tedavisi gerekebilir, hekim önerilerini takip edin.")
        
        suggestions.append("🥗 Akdeniz diyeti kalp sağlığı için en ideal beslenme modelidir.")
        suggestions.append("😴 Günde 7-8 saat kaliteli uyku uyuyun.")
        suggestions.append("🧘 Stres yönetimi için meditasyon veya yoga yapın.")
        suggestions.append("💧 Bol su için (günde 2-2.5 litre).")
        
        # Uyarıları göster
        if warnings:
            st.markdown("### ⚠️ Önemli Uyarılar")
            for warning in warnings:
                st.markdown(f'<div class="warning-box">{warning}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Önerileri göster
        st.markdown("### 📋 Sağlık Önerileri")
        for suggestion in suggestions:
            st.markdown(f'<div class="suggestion-box">{suggestion}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Risk azaltma simülasyonu
        st.markdown("### 📉 Risk Azaltma Simülasyonu")
        st.markdown("Aşağıdaki değişiklikler yapılırsa riskiniz nasıl değişebilir:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Şu anki durumunuz:")
            st.metric("Risk", f"%{risk*100:.1f}", delta=None)
        
        with col2:
            st.markdown("##### Öneriler uygulanırsa:")
            # Basit bir simülasyon: her iyileştirme %5-10 risk azaltabilir
            potential_reduction = 0
            if "ap_hi" in input_values and input_values["ap_hi"] > 140:
                potential_reduction += 15
            if "bmi" in input_values and input_values["bmi"] > 25:
                potential_reduction += 10
            if "smoke" in input_values and input_values["smoke"] == 1:
                potential_reduction += 20
            if "active" in input_values and input_values["active"] == 0:
                potential_reduction += 10
            if "cholesterol" in input_values and input_values["cholesterol"] > 1:
                potential_reduction += 8
            
            new_risk = max(5, risk*100 - potential_reduction)
            st.metric("Potansiyel Risk", f"%{new_risk:.1f}", delta=f"-%{potential_reduction:.0f}", delta_color="inverse")
        
        st.info(" Bu simülasyon tahminidir. Gerçek sonuçlar için hekim görüşü alın.")

st.markdown("---")
st.markdown("""
<center>
    <p style='color: gray; font-size: 12px;'>
         Bu uygulama sadece bilgilendirme amaçlıdır. Tıbbi tanı ve tedavi için mutlaka bir sağlık uzmanına başvurun.<br>
      
    
</center>
""", unsafe_allow_html=True)
