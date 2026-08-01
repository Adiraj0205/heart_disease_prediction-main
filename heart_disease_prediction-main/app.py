from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Handle JSON input (REST API request)
    if request.is_json:
        data = request.get_json()
        try:
            features = np.array([[
                float(data["age"]),
                float(data["sex"]),
                float(data["cp"]),
                float(data["trestbps"]),
                float(data["chol"]),
                float(data["fbs"]),
                float(data["restecg"]),
                float(data["thalach"]),
                float(data["exang"]),
                float(data["oldpeak"]),
                float(data["slope"]),
                float(data["ca"]),
                float(data["thal"])
            ]])
            
            prediction = model.predict(features)
            pred_val = int(prediction[0])
            result_text = "Heart Disease Detected" if pred_val == 1 else "No Heart Disease Detected"
            
            return jsonify({
                "status": "success",
                "prediction": result_text,
                "prediction_code": pred_val
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    # Handle Standard Form input
    age = int(request.form["age"])
    sex = int(request.form["sex"])
    cp = int(request.form["cp"])
    trestbps = int(request.form["trestbps"])
    chol = int(request.form["chol"])
    fbs = int(request.form["fbs"])
    restecg = int(request.form["restecg"])
    thalach = int(request.form["thalach"])
    exang = int(request.form["exang"])
    oldpeak = float(request.form["oldpeak"])
    slope = int(request.form["slope"])
    ca = int(request.form["ca"])
    thal = int(request.form["thal"])

    features = np.array([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak, slope, ca, thal
    ]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "High Risk of Heart Disease ❤️"
    else:
        result = "Low Risk of Heart Disease 💚"

    return render_template("result.html", prediction=result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    return predict()


if __name__ == "__main__":
    app.run(debug=True)