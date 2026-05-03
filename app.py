from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from groq import Groq
from backend.model import load_model

app = Flask(__name__)
models, mean_values, scaler, accuracy = load_model()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            age     = int(request.form.get("age", 45))
            sex     = int(request.form.get("sex", 1))
            genetic = int(request.form.get("genetic", 0))
            smoking = float(request.form.get("smoking", 0))
            fbs     = int(request.form.get("fbs", 0))
            cp      = int(request.form.get("cp", 4))

            t_map    = {"Dusuk": 110, "Orta": 130, "Yuksek": 160}
            c_map    = {"Dusuk": 180, "Orta": 220, "Yuksek": 290}
            trestbps = t_map.get(request.form.get("trestbps"), 130)
            chol     = c_map.get(request.form.get("chol"), 220)

            thalach = float(request.form.get("thalach_range", 145))
            oldpeak = float(request.form.get("oldpeak", 0.0))

            input_data = np.array([[
                age, sex, cp, trestbps, chol, fbs,
                mean_values['restecg'], thalach, 0,
                oldpeak, mean_values['slope'],
                mean_values['ca'], mean_values['thal']
            ]])
            scaled = scaler.transform(input_data)

            p_rf  = models['rf'].predict_proba(scaled)[0][1]  * 100
            p_xgb = models['xgb'].predict_proba(scaled)[0][1] * 100
            p_lr  = models['lr'].predict_proba(scaled)[0][1]  * 100
            ai_score = (p_rf * 0.45) + (p_xgb * 0.45) + (p_lr * 0.10)

            age_filter       = age / 55
            lifestyle_impact = (smoking * 22) + (genetic * 15) + (fbs * 12)
            final_risk       = (ai_score * 0.5 * age_filter) + (lifestyle_impact * (age / 50))
            final_risk       = min(max(final_risk, 3.5), 98.8)

            return render_template("index.html",
                risk     = round(final_risk, 1),
                accuracy = accuracy,
                age      = age,
                sex      = sex,
                genetic  = genetic,
                smoking  = smoking,
                fbs      = fbs,
                cp       = cp,
                trestbps = trestbps,
                chol     = chol,
                thalach  = round(thalach, 1),
                oldpeak  = round(oldpeak, 1),
            )
        except Exception as e:
            return render_template("index.html", error=str(e))
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data     = request.get_json()
        system   = data.get("system", "")
        messages = data.get("messages", [])

        full_messages = [{"role": "system", "content": system}] + messages

        response = client.chat.completions.create(
            model = "llama-3.3-70b-versatile",
            messages   = full_messages,
            max_tokens = 1024,
        )

        return jsonify({
            "content": [{"text": response.choices[0].message.content}]
        })

    except Exception as e:
        print("HATA:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True, use_reloader=False)