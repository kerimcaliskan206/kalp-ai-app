from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials, firestore, auth
from groq import Groq
from backend.model import load_model
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "kalp-ai-secret-2024")

# Firebase init
import json
firebase_key = os.environ.get("FIREBASE_KEY_JSON")
if firebase_key:
    cred = credentials.Certificate(json.loads(firebase_key))
else:
    cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

models, mean_values, scaler, accuracy = load_model()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_current_user():
    return session.get("user")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        id_token = request.form.get("idToken")
        try:
            decoded = auth.verify_id_token(id_token)
            session["user"] = {
                "uid":   decoded["uid"],
                "email": decoded.get("email", ""),
                "name":  decoded.get("name", decoded.get("email", "Kullanıcı"))
            }
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 401
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/gecmis")
def gecmis():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))
    analizler = []
    try:
        docs = db.collection("analizler")\
                 .where("uid", "==", user["uid"])\
                 .order_by("tarih", direction=firestore.Query.DESCENDING)\
                 .limit(20)\
                 .stream()
        for doc in docs:
            d = doc.to_dict()
            d["id"] = doc.id
            analizler.append(d)
    except Exception as e:
        print("Geçmiş hatası:", e)
    return render_template("gecmis.html", analizler=analizler, user=user)

@app.route("/", methods=["GET", "POST"])
def home():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

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
            thalach  = float(request.form.get("thalach_range", 145))
            oldpeak  = float(request.form.get("oldpeak", 0.0))

            input_data = np.array([[
                age, sex, cp, trestbps, chol, fbs,
                mean_values['restecg'], thalach, 0,
                oldpeak, mean_values['slope'],
                mean_values['ca'], mean_values['thal']
            ]])
            scaled   = scaler.transform(input_data)
            p_rf     = models['rf'].predict_proba(scaled)[0][1]  * 100
            p_xgb    = models['xgb'].predict_proba(scaled)[0][1] * 100
            p_lr     = models['lr'].predict_proba(scaled)[0][1]  * 100
            ai_score = (p_rf * 0.45) + (p_xgb * 0.45) + (p_lr * 0.10)

            age_filter       = age / 55
            lifestyle_impact = (smoking * 22) + (genetic * 15) + (fbs * 12)
            final_risk       = (ai_score * 0.5 * age_filter) + (lifestyle_impact * (age / 50))
            final_risk       = min(max(final_risk, 3.5), 98.8)
            final_risk       = round(final_risk, 1)

            # Firebase'e kaydet
            try:
                db.collection("analizler").add({
                    "uid":      user["uid"],
                    "email":    user["email"],
                    "tarih":    datetime.utcnow(),
                    "risk":     final_risk,
                    "accuracy": accuracy,
                    "age": age, "sex": sex, "genetic": genetic,
                    "smoking": smoking, "fbs": fbs, "cp": cp,
                    "trestbps": trestbps, "chol": chol,
                    "thalach": round(thalach,1), "oldpeak": round(oldpeak,1)
                })
            except Exception as e:
                print("Kayıt hatası:", e)

            return render_template("index.html",
                risk=final_risk, accuracy=accuracy,
                age=age, sex=sex, genetic=genetic,
                smoking=smoking, fbs=fbs, cp=cp,
                trestbps=trestbps, chol=chol,
                thalach=round(thalach,1), oldpeak=round(oldpeak,1),
                user=user
            )
        except Exception as e:
            return render_template("index.html", error=str(e), user=user)
    return render_template("index.html", user=user)

@app.route("/api/chat", methods=["POST"])
def chat():
    if not get_current_user():
        return jsonify({"error": "Giriş gerekli"}), 401
    try:
        data     = request.get_json()
        system   = data.get("system", "")
        messages = data.get("messages", [])
        full_messages = [{"role": "system", "content": system}] + messages
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=full_messages,
            max_tokens=1024,
        )
        return jsonify({"content": [{"text": response.choices[0].message.content}]})
    except Exception as e:
        print("HATA:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)