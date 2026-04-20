from flask import Flask, render_template, request
import numpy as np
from backend.model import load_model

app = Flask(__name__)
model, mean_values, scaler = load_model()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Formdan gelen kullanıcı verileri
            age = int(request.form.get("age", 40))
            sex = int(request.form.get("sex", 1))
            cp = int(request.form.get("cp", 4)) # Göğüs ağrısı tipi
            trestbps = int(request.form.get("trestbps", 120)) # Tansiyon
            chol = int(request.form.get("chol", 200)) # Kolesterol
            fbs = int(request.form.get("fbs", 0)) # Şeker
            exang = int(request.form.get("exang", 0)) # Egzersiz ağrısı
            thalach = float(request.form.get("thalach", 80)) # Nabız
            oldpeak = float(request.form.get("oldpeak", 0)) # Ağrı şiddeti (0-5)

            # 1. ADIM: Ham Tahmin (Modelin Makine Öğrenmesi Tahmini)
            input_array = np.array([[
                age, sex, cp, trestbps, chol, fbs,
                float(mean_values.get("restecg", 0)),
                thalach, exang, oldpeak,
                float(mean_values.get("slope", 1)),
                float(mean_values.get("ca", 0)),
                float(mean_values.get("thal", 2))
            ]])

            scaled_input = scaler.transform(input_array)
            prob = model.predict_proba(scaled_input)
            base_risk = prob[0][1] * 100 # Modelin verdiği temel risk

            # 2. ADIM: Mükemmel Tahmin İçin "Hassas Ayar" (Fine-Tuning)
            # Model bazen veriyi küçümseyebilir, burada tıbbi mantığı devreye sokuyoruz.
            
            final_risk = base_risk

            # Eğer ağrı şiddeti (oldpeak) 3'ten büyükse riski artır
            if oldpeak >= 4:
                final_risk += 20
            elif oldpeak >= 2:
                final_risk += 10

            # Eğer göğüs ağrısı (cp) şiddetliyse (değeri 1 ise modelde risklidir)
            if cp == 1:
                final_risk += 15
            
            # Şeker ve yüksek tansiyon kombinasyonu varsa ekstra risk
            if fbs == 1 and trestbps >= 150:
                final_risk += 10

            # Riski %100 ile sınırla
            final_risk = min(max(final_risk, 0), 100)

            # 3. ADIM: Halk Ağzıyla Yorumlama
            yorumlar = []
            if final_risk > 75:
                yorumlar.append("⚠️ Dikkat: Risk oranınız yüksek görünüyor. En kısa sürede bir kardiyoloji uzmanına görünmenizi öneririz.")
            elif final_risk > 40:
                yorumlar.append("🟡 Uyarı: Bazı değerleriniz sınırda. Yaşam tarzınıza, beslenmenize dikkat etmeli ve düzenli kontrol yaptırmalısınız.")
            else:
                yorumlar.append("✅ Güzel Haber: Analiz sonuçlarına göre kalp riskiniz düşük görünüyor. Sağlıklı yaşamaya devam edin!")

            # Özel durum notları
            if oldpeak >= 3:
                yorumlar.append("• Belirttiğiniz ağrı şiddeti kalp zorlanmasına işaret ediyor olabilir.")
            if fbs == 1:
                yorumlar.append("• Yüksek şeker kalp damarlarını yorabilir, şeker takibi yapmayı unutmayın.")

            return render_template("index.html", risk=round(final_risk, 1), yorumlar=yorumlar)

        except Exception as e:
            print(f"Hata detayı: {e}")
            return render_template("index.html", risk=0, yorumlar=["Analiz yapılırken bir hata oluştu."])
            
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)