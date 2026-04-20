from flask import Flask, render_template, request
from backend.model import load_model

app = Flask(__name__)

# model yükle
model, mean_values, scaler = load_model()


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":
        try:
            # 🔥 INPUT
            age = int(request.form.get("age", 40))
            sex = int(request.form.get("sex", 1))
            cp = int(request.form.get("cp", 3))
            trestbps = int(request.form.get("trestbps", 120))
            chol = int(request.form.get("chol", 200))
            exang = int(request.form.get("exang", 0))

            thalach = float(request.form.get("thalach", 120))
            oldpeak = float(request.form.get("oldpeak", 2))

            # 🔥 MODEL INPUT
            input_data = [[
                age, sex, cp, trestbps, chol,
                float(mean_values["fbs"]),
                float(mean_values["restecg"]),
                thalach,
                exang,
                oldpeak,
                float(mean_values["slope"]),
                float(mean_values["ca"]),
                float(mean_values["thal"])
            ]]

            input_data = scaler.transform(input_data)

            # 🔥 MODEL SONUÇ
            prob = model.predict_proba(input_data)
            model_risk = prob[0][1] * 100

            # 🔥 RİSK FAKTÖRLERİ SAY
            risk_factors = 0

            if chol > 240:
                risk_factors += 1

            if trestbps > 140:
                risk_factors += 1

            if exang == 1:
                risk_factors += 1

            if thalach > 180:
                risk_factors += 1

            if oldpeak >= 4:
                risk_factors += 1

            if age > 60:
                risk_factors += 1

            # 🔥 TEMEL RİSK (model ağırlıklı)
            risk = model_risk * 0.6

            # 🔥 AKILLI ARTIŞ (non-linear)
            if risk_factors == 1:
                risk += 15
            elif risk_factors == 2:
                risk += 30
            elif risk_factors == 3:
                risk += 45
            elif risk_factors >= 4:
                risk += 60

            # 🔥 SENİN İSTEDİĞİN ÖZEL DURUM
            if thalach > 180 and oldpeak >= 4:
                risk = max(risk, 50)

            # 🔥 SINIRLA
            risk = min(max(risk, 0), 100)

            # 🔥 YORUMLAR
            yorumlar = []

            if chol > 240:
                yorumlar.append("Kolesterol yüksek")

            if trestbps > 140:
                yorumlar.append("Tansiyon yüksek")

            if exang == 1:
                yorumlar.append("Egzersizde ağrı var")

            if thalach > 180:
                yorumlar.append("Kalp atış hızı çok yüksek")

            if oldpeak >= 4:
                yorumlar.append("Ağrı seviyesi çok yüksek")

            if age > 60:
                yorumlar.append("Yaş faktörü riski artırıyor")

            # 🔥 GENEL YORUM
            if risk < 30:
                yorumlar.append("Genel risk düşük")
            elif risk < 70:
                yorumlar.append("Orta risk, dikkat edilmeli")
            else:
                yorumlar.append("Yüksek risk, doktora danışılmalı")

            return render_template(
                "index.html",
                risk=round(risk, 2),
                yorumlar=yorumlar
            )

        except Exception as e:
            print("HATA:", e)
            return render_template(
                "index.html",
                risk=0,
                yorumlar=["Bir hata oluştu"]
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)