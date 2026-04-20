from flask import Flask, render_template, request
from backend.model import load_model

app = Flask(__name__)

model, mean_values, scaler = load_model()

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":
        try:
            # 🔥 SAFE INPUT
            age = int(request.form.get("age", 40))
            sex = int(request.form.get("sex", 1))
            cp = int(request.form.get("cp", 3))
            trestbps = int(request.form.get("trestbps", 120))
            chol = int(request.form.get("chol", 200))
            exang = int(request.form.get("exang", 0))

            thalach = float(request.form.get("thalach", 150))
            oldpeak = float(request.form.get("oldpeak", 1.0))

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

            prob = model.predict_proba(input_data)
            risk = prob[0][1] * 100
            yorumlar = []

            if chol > 240:
                yorumlar.append("Kolesterol yüksek")

            if trestbps > 140:
                yorumlar.append("Tansiyon yüksek")

            if exang == 1:
                yorumlar.append("Egzersizde ağrı var")

            if thalach < 140:
                yorumlar.append("Kalp atış hızı düşük")

            if oldpeak > 2:
                yorumlar.append("Efor sonrası ağrı yüksek")

            return render_template("index.html", risk=round(risk, 2), yorumlar=yorumlar)

        except Exception as e:
            return f"<h2>HATA:</h2><pre>{e}</pre>"

    return render_template("index.html")


if __name__ == "__main__":
    import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))