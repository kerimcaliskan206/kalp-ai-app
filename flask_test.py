from flask import Flask, render_template, request
import numpy as np
from backend.model import load_model

app = Flask(__name__)
model, mean_values, scaler = load_model()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Verileri al
            age = int(request.form.get("age", 40))
            sex = int(request.form.get("sex", 1))
            cp = int(request.form.get("cp", 4)) 
            trestbps = int(request.form.get("trestbps", 120))
            chol = int(request.form.get("chol", 200))
            fbs = int(request.form.get("fbs", 0))
            exang = int(request.form.get("exang", 0))
            thalach = float(request.form.get("thalach", 150)) # Max nabız (Genelde 150 civarıdır)
            oldpeak = float(request.form.get("oldpeak", 0))

            # 13 Parametrelik Giriş Dizisi
            input_array = np.array([[
                age, sex, cp, trestbps, chol, fbs,
                float(mean_values.get("restecg", 0)),
                thalach, exang, oldpeak,
                float(mean_values.get("slope", 1)),
                float(mean_values.get("ca", 0)),
                float(mean_values.get("thal", 2))
            ]])

            # Standartlaştırma ve Olasılık Tahmini
            scaled_input = scaler.transform(input_array)
            prob = model.predict_proba(scaled_input)
            
            # Modelin saf tahmini (% olarak)
            base_risk = prob[0][1] * 100 

            # HASSAS AYAR (Fine-Tuning): 
            # Modeli çok saptırmadan sadece uç değerleri kontrol ediyoruz.
            final_risk = base_risk

            # Eğer ağrı şiddeti (oldpeak) çok yüksekse küçük bir dokunuş yap
            if oldpeak > 3.5:
                final_risk += 10 # Eskiden 20'ydi, çok fazlaydı.
            
            # Eğer göğüs ağrısı (cp) şiddetliyse ve yaş 50+ ise
            if cp == 1 and age > 50:
                final_risk += 5

            # Riski mantıklı sınırlar içinde tut (%0 - %100)
            final_risk = min(max(final_risk, 0), 99.9)

            # Yorumları yumuşatalım
            yorumlar = []
            if final_risk > 80:
                yorumlar.append("Sonuçlar riskli görünüyor. Lütfen ihmal etmeden bir uzmana danışın.")
            elif final_risk > 45:
                yorumlar.append("Orta düzeyde bir risk görüldü. Sağlıklı beslenme ve egzersizle bu durumu iyileştirebilirsiniz.")
            else:
                yorumlar.append("Kalp sağlığınız şu anki verilere göre gayet iyi görünüyor.")

            # Kritik uyarılar (Sadece gerçekten gerekliyse)
            if trestbps > 160:
                yorumlar.append("• Tansiyonunuz oldukça yüksek, dinlenmeniz gerekebilir.")

            return render_template("index.html", risk=round(final_risk, 1), yorumlar=yorumlar)

        except Exception as e:
            return render_template("index.html", risk=0, yorumlar=["Analiz sırasında bir hata oluştu."])
            
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)