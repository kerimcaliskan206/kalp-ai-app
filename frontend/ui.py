import streamlit as st

def run_app(model, acc):

    # CSS YÜKLE
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.set_page_config(
        page_title="Kalp Sağlığı",
        page_icon="❤️",
        layout="centered"
    )

    # 🔥 YENİ BAŞLIK (CSS ile bağlı)
    st.markdown("""
    <div class="header-box">
        <h1>❤️ Kalp Sağlığı Analizi</h1>
        <p>Riskinizi kolayca öğrenin</p>
    </div>
    """, unsafe_allow_html=True)

    # MODEL DOĞRULUK
    st.markdown(f"""
    <div class="card">
    📊 Model doğruluğu: %{acc*100:.2f}
    </div>
    """, unsafe_allow_html=True)

    # ---------------- FORM ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    age = st.slider("Yaş", 20, 100, 40)

    sex = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
    sex = 1 if sex == "Erkek" else 0

    cp_options = {
        "Hiç ağrı yok": 4,
        "Hafif ağrı": 3,
        "Orta seviye ağrı": 2,
        "Şiddetli ağrı": 1
    }
    cp = cp_options[st.selectbox("Göğüs ağrısı", list(cp_options.keys()))]

    trestbps = 150 if st.selectbox("Tansiyon", ["Normal", "Yüksek"]) == "Yüksek" else 120
    chol = 260 if st.selectbox("Kolesterol", ["Normal", "Yüksek"]) == "Yüksek" else 200

    exang = st.selectbox("Egzersizde ağrı", ["Hayır", "Evet"])
    exang = 1 if exang == "Evet" else 0

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TAHMİN ----------------
    if st.button("🔍 Tahmin Et", use_container_width=True):

        with st.spinner("Analiz yapılıyor..."):

            input_data = [[
                age, sex, cp, trestbps, chol,
                0, 1, 150, exang, 1.0, 2, 0, 3
            ]]

            prob = model.predict_proba(input_data)
            risk = prob[0][1] * 100

        # 🔥 YENİ SONUÇ KARTI
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        st.subheader("📊 Sonuç")
        st.write(f"Risk oranı: %{risk:.2f}")

        if risk < 30:
            yorum = "Genel olarak sağlıklı görünüyorsunuz."
            renk = "#00ff88"
            text = "🟢 Düşük risk"
        elif risk < 70:
            yorum = "Bazı risk faktörleri mevcut, dikkatli olun."
            renk = "#ffcc00"
            text = "🟡 Orta risk"
        else:
            yorum = "Risk yüksek, bir uzmana danışmanız önerilir."
            renk = "#ff4d4d"
            text = "🔴 Yüksek risk"

        st.markdown(f"""
        <div style="background:{renk}20;padding:12px;border-radius:10px;text-align:center;color:{renk};">
        {text}
        </div>
        """, unsafe_allow_html=True)

        st.write(yorum)

        st.info("Bu bir yapay zeka tahminidir.")

        # 🔥 ALT TEXT (CSS ile bağlı)
        st.markdown(
            '<p class="small-text">Bu sistem makine öğrenmesi modeli kullanarak tahmin yapar.</p>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)