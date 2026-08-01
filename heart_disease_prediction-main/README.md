# AI-ML Assignment 10 - Heart Disease Prediction & MLOps Deployment

An end-to-end Machine Learning web application designed to assess heart disease risk from clinical health parameters. Built with **Python**, **Flask**, and **Scikit-Learn**, featuring a modern responsive UI, REST API endpoint support, and cloud deployment configuration for Render.

---

## 🌟 Highlights & Features

- **Clinical Parameter Input**: Simple interface to input 13 standard clinical parameters.
- **REST API Endpoint**: Supports programmatic prediction requests via JSON (`POST /api/predict`).
- **Interactive Modern UI**: Designed with clean glassmorphism styling, micro-animations, and visual feedback cards.
- **Machine Learning Pipeline**: Trained using model evaluation metrics (Random Forest / Logistic Regression / Decision Tree) with automated model serialization (`model.pkl`).
- **Cloud Ready**: Prepared with `gunicorn`, `Procfile`, and `render.yaml` for seamless deployment on Render.

---

## 🚀 Live Web Application

🔗 **Render Live Deployment URL:**  
`https://heart-disease-prediction-main-muux.onrender.com`

---

## 🛠️ Technology Stack

- **Backend Framework**: Python 3, Flask
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **WSGI Server**: Gunicorn
- **Frontend**: HTML5, CSS3 (Modern Glassmorphism & Gradient styling), JavaScript

---

## 📂 Repository Structure

```text
HeartDiseaseDeployment/
│
├── app.py              # Flask Web Application & REST API implementation
├── train_model.py      # Data preprocessing, model training & evaluation script
├── model.pkl           # Trained machine learning model binary
├── heart.csv           # UCI Heart Disease Dataset
├── requirements.txt    # Python package dependencies
├── Procfile            # Render web process definition
├── render.yaml         # Render blueprint configuration
├── README.md           # Documentation & Project Report
├── .gitignore          # Git exclusion rules
│
├── templates/
│   ├── index.html      # Main clinical input form page
│   └── result.html     # Prediction assessment result view
│
└── static/
    └── style.css       # Visual design system & modern responsive styles
```

---

## ☁️ How to Deploy to Render

To host this repository live on [Render](https://render.com/):

1. Push this repository to **GitHub** (ensure repository is **Public** as per assignment rules).
2. Log into [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** -> **Web Service**.
4. Connect your GitHub repository (`heart_disease_prediction`).
5. Configure deployment settings:
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
6. Click **Create Web Service**. Render will automatically build and deploy your app.
7. Paste your live Render URL into the `README.md` and submission Google Form.

---

## ⚙️ Local Setup & Running

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/heart_disease_prediction.git
cd heart_disease_prediction
```

### 2. Set Up Virtual Environment
```bash
# Create environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Model & Save Binary
```bash
python train_model.py
```

### 5. Launch Local Flask Server
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your web browser.

---

## 🔌 REST API Usage

You can query the deployed prediction service using JSON payloads:

### Request Example (`POST /api/predict` or `/predict`)
```json
{
  "age": 57,
  "sex": 1,
  "cp": 2,
  "trestbps": 130,
  "chol": 236,
  "fbs": 0,
  "restecg": 1,
  "thalach": 174,
  "exang": 0,
  "oldpeak": 0.0,
  "slope": 1,
  "ca": 1,
  "thal": 2
}
```

### Response Example
```json
{
  "status": "success",
  "prediction": "No Heart Disease Detected",
  "prediction_code": 0
}
```

---

## 📝 Conclusion & MLOps Insights

### Model Performance
During model comparison, multiple algorithms (Logistic Regression, Decision Tree, Random Forest, KNN, SVM) were evaluated on 80/20 train-test split data. The top-performing model achieved reliable classification accuracy on test patient records.

### Deployment Challenges
Key challenges included handling web request payload differences (handling both HTML Form submissions and JSON REST API bodies), ensuring python dependency versions matched across local environments and cloud containers, and configuring Gunicorn runtime settings for cloud hosting on Render.

### Importance of MLOps
MLOps plays a critical role in bridging the gap between machine learning model experimentation and production deployment. By standardizing model artifact serialization (`joblib`), maintaining automated dependency tracking (`requirements.txt`), and establishing continuous deployment pipelines via GitHub and cloud web services, ML solutions become reproducible, scalable, and readily accessible for real-world application.

---

## ⚠️ Disclaimer
This application is created solely for educational and academic purposes. The outputs provided by the model must not be interpreted as professional medical advice or clinical diagnosis.
