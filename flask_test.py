from flask import Flask, render_template, request
import numpy as np
from backend.model import load_model

app = Flask(__name__)
model, mean_values, scaler = load_model()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Form verileri
            age = int(request.form.get("age", 40))
            sex = int(request.form.get("sex", 1))
            cp = int(request.form.get("cp", 4)) 
            trestbps = int(request.form.get("trestbps", 120))
            chol = int(request.form.get("chol", 200))
            fbs = int(request.form.get("fbs", 0))
            exang = int(request.form.get("exang", 0))
            thalach = float(request.form.get("thalach", 80))
            oldpeak = float(request.form.get("oldpeak", 0))

            # 1. AI Tahmini
            input_array = np.array([[
                age, sex, cp, trestbps, chol, fbs,
                float(mean_values.get("restecg", 0)),
                thalach, exang, oldpeak,
                float(mean_values.get("slope", 1)),
                float(mean_values.get("ca", 0)),
                float(mean_values.get("thal", 2))
            ]])

            scaled_input = scaler.transform(input_array)
            ai_prob = model.predict_proba(scaled_input)[0][1] * 100

            # 2. Dinamik Risk Filtresi (Yaş Duyarlı)
            # Gençlerde riskin bu kadar kolay fırlamasını engelliyoruz.
            age_factor = 1.0
            if age < 30:
                age_factor = 0.5  # 30 yaş altı için risk etkisini yarıya indir
            elif age < 45:
                age_factor = 0.8

            # Tıbbi puanlama (Kritik değerler)
            medical_points = 0
            if cp == 1: medical_points += 20
            if exang == 1: medical_points += 15
            if oldpeak >= 3: medical_points += 20
            if trestbps >= 150: medical_points += 10
            
            # Final Hesaplama: AI tahmini ile tıbbi mantığı yaş faktörüyle çarpıyoruz
            # Gençse, AI'nın veya bizim verdiğimiz puanların etkisi azalır.
            final_risk = (ai_prob * 0.5) + (medical_points * age_factor)

            # Mantıksal Sınırlar
            if age < 25 and medical_points < 20:
                final_risk = min(final_risk, 15.0) # Genç ve ağır belirtisi olmayana yüksek risk verme

            final_risk = min(max(final_risk, 5.2), 99.1)

            yorumlar = []
            if final_risk > 70:
                yorumlar.append("⚠️ Riskli seviye. Bir uzmana danışmanızda fayda var.")
            elif final_risk > 35:
                yorumlar.append("🟡 Orta seviye. Yaşam tarzınıza dikkat etmelisiniz.")
            else:
                yorumlar.append("✅ Düşük risk. Kalp sağlığınız iyi görünüyor.")

            return render_template("index.html", risk=round(final_risk, 1), yorumlar=yorumlar)

        except:
            return render_template("index.html", risk=0, yorumlar=["Hata oluştu."])
            
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)