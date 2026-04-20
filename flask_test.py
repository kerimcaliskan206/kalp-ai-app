from flask import Flask, render_template, request
import numpy as np
from backend.model import load_model

app = Flask(__name__)
model, mean_values, scaler = load_model()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Form verilerini alıyoruz
            age = int(request.form.get("age", 40))
            sex = int(request.form.get("sex", 1))
            cp = int(request.form.get("cp", 4))
            trestbps = int(request.form.get("trestbps", 120))
            chol = int(request.form.get("chol", 200))
            fbs = int(request.form.get("fbs", 0))
            exang = int(request.form.get("exang", 0))
            thalach = float(request.form.get("thalach", 80))
            oldpeak = float(request.form.get("oldpeak", 0))

            # 13 özellikli giriş (Modelin tam istediği format)
            input_array = np.array([[
                age, sex, cp, trestbps, chol, fbs,
                float(mean_values.get("restecg", 0)),
                thalach, exang, oldpeak,
                float(mean_values.get("slope", 1)),
                float(mean_values.get("ca", 0)),
                float(mean_values.get("thal", 2))
            ]])

            # Ölçeklendirme ve Tahmin
            scaled_input = scaler.transform(input_array)
            prob = model.predict_proba(scaled_input)
            risk = prob[0][1] * 100

            # Halk ağzıyla sonuç yorumlama
            yorumlar = []
            if risk > 65:
                yorumlar.append("Sonuçlar biraz yüksek, bir doktora görünmeniz iyi olabilir.")
            elif risk > 30:
                yorumlar.append("Bazı değerleriniz sınırda, kendinizi yormamaya çalışın.")
            else:
                yorumlar.append("Kalp sağlığınız şu an gayet iyi görünüyor.")

            return render_template("index.html", risk=round(risk, 1), yorumlar=yorumlar)
        except:
            return render_template("index.html", risk=0, yorumlar=["Analiz yapılamadı."])
            
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)