# 🛡️ AI/ML-Based Fraud Detection System

An end-to-end **AI/ML Fraud Detection System** that analyzes financial transaction behavior, predicts the probability of fraud in real time, assigns a risk score/risk level, stores prediction history, and provides a monitoring dashboard.

The project uses a **tuned XGBoost classifier** trained on a 10,000-transaction fraud dataset and exposes the trained model through a **FastAPI REST API**. A **Streamlit dashboard** provides real-time prediction and monitoring.

---

## 📌 Project Overview

Financial organizations process thousands of transactions every day. Detecting fraudulent transactions manually is slow, difficult to scale, and may miss complex transaction patterns.

This project provides an automated fraud detection pipeline that:

- Cleans and validates transaction data
- Performs exploratory data analysis
- Handles class imbalance using sampling techniques
- Engineers additional fraud-related features
- Compares multiple machine learning algorithms
- Tunes an XGBoost model
- Generates fraud probabilities and risk scores
- Provides real-time fraud prediction through REST APIs
- Stores prediction results in a database
- Provides a Streamlit monitoring dashboard
- Includes input validation and exception handling

---

## 🎯 Problem Statement

Build an AI/ML-based fraud detection solution capable of identifying suspicious financial transactions and helping organizations reduce potential financial losses.

The system should analyze transaction characteristics, detect suspicious behavior, assign risk levels, and make predictions available through a real-time API and monitoring dashboard.

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────────┐
                         │   User / Transaction     │
                         │                          │
                         │ Amount                   │
                         │ Transaction Hour         │
                         │ Merchant Category        │
                         │ Foreign Transaction      │
                         │ Location Mismatch        │
                         │ Device Trust Score       │
                         │ Velocity Last 24h        │
                         │ Cardholder Age           │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │       Streamlit          │
                         │       Dashboard          │
                         └────────────┬─────────────┘
                                      │ REST API
                                      ▼
                         ┌──────────────────────────┐
                         │        FastAPI           │
                         │      /predict API        │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │   Feature Engineering    │
                         │                          │
                         │ log_amount               │
                         │ high_value_transaction   │
                         │ low_device_trust         │
                         │ high_velocity            │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │    Tuned XGBoost Model   │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────┼────────────┐
                         ▼            ▼            ▼
                    Probability   Risk Score   Prediction
                         │            │            │
                         └────────────┼────────────┘
                                      ▼
                              LOW / MEDIUM / HIGH
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ SQLite / PostgreSQL      │
                         │ Prediction History       │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ Monitoring Dashboard     │
                         │ Fraud / Risk / Patterns  │
                         │ Model Performance        │
                         └──────────────────────────┘
```

---

# 📊 Dataset Details

The project was trained using a **10,000-transaction credit-card fraud dataset**.

### Dataset size

- Total transactions: **10,000**
- Legitimate transactions: **9,849**
- Fraudulent transactions: **151**
- Fraud rate: **1.51%**

### Input attributes

| Feature | Description |
|---|---|
| `transaction_id` | Unique transaction identifier; excluded from model training |
| `amount` | Transaction amount |
| `transaction_hour` | Hour at which the transaction occurred, 0–23 |
| `merchant_category` | Merchant/business category |
| `foreign_transaction` | Whether the transaction is foreign |
| `location_mismatch` | Whether transaction location differs from expected location |
| `device_trust_score` | Trust score of the device, 0–100 |
| `velocity_last_24h` | Number of transactions in the previous 24 hours |
| `cardholder_age` | Cardholder age |
| `is_fraud` | Target variable: 1 = fraud, 0 = legitimate |

> **Note:** This implementation does not require users to enter technical `V1`, `V2`, `V3`, etc. features. The deployed model accepts human-readable transaction information.

---

# 🧹 Data Preprocessing

The following preprocessing steps were implemented:

### 1. Data validation

The notebook validates the expected dataset columns before training.

### 2. Missing-value handling

- Numerical missing values are replaced using the median.
- Categorical missing values are replaced using the mode.

### 3. Duplicate handling

Duplicate transaction records are removed before model training.

### 4. Outlier analysis

Transaction amount outliers are analyzed using the IQR method.

Fraud-related extreme values are not blindly deleted because unusual transaction behavior can be an important fraud signal.

### 5. Feature engineering

Additional features are generated:

```text
log_amount
high_value_transaction
low_device_trust
high_velocity
```

### 6. Categorical encoding

`merchant_category` is converted using One-Hot Encoding.

### 7. Numerical scaling

Numerical features are standardized using `StandardScaler`.

### 8. Class imbalance

Because fraud represents only a small portion of the dataset, the project evaluates:

- SMOTE
- Random Over Sampling
- Random Under Sampling

The final XGBoost model uses **SMOTE on the training data only**.

---

# 🤖 Machine Learning Algorithms

The project compares multiple approaches.

### Supervised Learning

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Unsupervised / Anomaly Detection

- Isolation Forest

### Optional

- Neural Network can be added using TensorFlow/Keras.

---

# 🔧 XGBoost Hyperparameter Tuning

RandomizedSearchCV is used to optimize the XGBoost model.

Parameters considered include:

```text
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
min_child_weight
```

The tuning objective is based on **F1-score**, which is useful for an imbalanced fraud-detection problem.

---

# 📈 Model Evaluation

The final tuned XGBoost model was evaluated on a held-out test set.

| Metric | Result |
|---|---:|
| Precision | **0.9677** |
| Recall | **1.0000** |
| F1-Score | **0.9836** |
| ROC-AUC | **1.0000** |
| PR-AUC | **1.0000** |
| Accuracy | **0.9995** |

### Metric explanation

**Precision**

Measures how many transactions predicted as fraud were actually fraudulent.

**Recall**

Measures how many actual fraudulent transactions were detected.

**F1-Score**

Balances precision and recall.

**ROC-AUC**

Measures the model's ability to distinguish fraud from legitimate transactions across classification thresholds.

**PR-AUC**

Provides useful information for highly imbalanced classification problems.

**Accuracy**

Measures the overall percentage of correct predictions.

> These results are specific to the uploaded dataset and the selected train/test split. They should not be interpreted as guaranteed real-world production performance.

---

# 💰 Fraud Probability and Risk Score

The model generates a probability between 0 and 1.

Example:

```text
Fraud Probability = 0.87
```

This is converted into a risk score:

```text
Risk Score = 87.00 / 100
```

Risk levels are assigned as:

| Fraud Probability | Risk Level |
|---:|---|
| `< 30%` | 🟢 LOW |
| `30% – < 70%` | 🟡 MEDIUM |
| `>= 70%` | 🔴 HIGH |

The API also returns:

```json
{
  "prediction": 1,
  "fraud_probability": 0.87,
  "risk_score": 87.0,
  "risk_level": "HIGH",
  "latency_ms": 4.21
}
```

---

# 🚀 REST API Documentation

The backend is implemented using **FastAPI**.

## Start the API

```bash
uvicorn app.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### `GET /`

Returns basic API information.

### `GET /health`

Checks whether the API and model are available.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "XGBoost"
}
```

### `POST /predict`

Generates a real-time fraud prediction.

#### Request

```json
{
  "amount": 850.50,
  "transaction_hour": 2,
  "merchant_category": "Electronics",
  "foreign_transaction": 1,
  "location_mismatch": 1,
  "device_trust_score": 25,
  "velocity_last_24h": 12,
  "cardholder_age": 29
}
```

#### Response

```json
{
  "transaction_id": 1,
  "prediction": 1,
  "fraud_probability": 0.87,
  "risk_score": 87.0,
  "risk_level": "HIGH",
  "latency_ms": 4.21
}
```

### `GET /transactions`

Returns recent prediction history.

Optional parameters:

```text
?limit=100
```

Fraud-only results:

```text
?fraud_only=true
```

### `GET /metrics`

Returns monitoring statistics and trained-model metrics.

Example:

```json
{
  "total_transactions": 100,
  "fraud_transactions": 5,
  "fraud_rate": 5.0,
  "average_risk_score": 24.5,
  "average_latency_ms": 3.2,
  "model_metrics": {
    "Precision": 0.9677,
    "Recall": 1.0,
    "F1": 0.9836,
    "ROC-AUC": 1.0,
    "PR-AUC": 1.0
  }
}
```

---

# 🖥️ Monitoring Dashboard

The Streamlit dashboard provides:

### 1. Transaction Metrics

- Total transactions
- Fraud transactions
- Fraud rate
- Average risk score
- Average API latency

### 2. Real-Time Fraud Prediction

Users can enter:

- Transaction amount
- Transaction hour
- Merchant category
- Foreign transaction
- Location mismatch
- Device trust score
- Transactions in last 24 hours
- Cardholder age

The dashboard sends the transaction to FastAPI and displays the result.

### 3. Fraud Transactions

Displays transactions classified as fraudulent.

### 4. Risk Level Distribution

Visualizes:

```text
LOW
MEDIUM
HIGH
```

### 5. Transaction Risk Pattern

Shows risk score behavior over time.

### 6. Fraud Probability Distribution

Displays the distribution of model fraud probabilities.

### 7. Merchant Category Fraud Pattern

Displays observed fraud rates across merchant categories.

### 8. Model Performance

Displays:

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC

---

# 🗄️ Database

The application uses **SQLAlchemy** for database access.

### Default

SQLite is used so the project can run immediately:

```text
fraud_predictions.db
```

Prediction records include:

- Transaction details
- Prediction
- Fraud probability
- Risk score
- Risk level
- API latency
- Timestamp

### PostgreSQL

The application can also use PostgreSQL by setting:

```text
DATABASE_URL=postgresql+psycopg2://username:password@localhost:5432/fraud_db
```

---

# 📁 Project Structure

```text
FRAUD_DETECTION_SYSTEM/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── database.py
│   ├── main.py
│   ├── model.py
│   └── schemas.py
│
├── artifacts/
│   ├── fraud_detection_xgboost.pkl
│   ├── fraud_model_features.pkl
│   └── fraud_model_metadata.pkl
│
├── dashboard.py
├── fraud_predictions.db
├── requirements.txt
├── README.md
└── AI_Fraud_Detection_Training.ipynb
```

---

# 🧩 Model Serialization

The trained model is stored using Joblib.

```text
fraud_detection_xgboost.pkl
```

The feature list is stored separately:

```text
fraud_model_features.pkl
```

Model metadata and evaluation information:

```text
fraud_model_metadata.pkl
```

The main model artifact contains the preprocessing pipeline and trained XGBoost model so that inference uses the same preprocessing logic as training.

---

# 🛡️ Input Validation & Exception Handling

FastAPI/Pydantic validates API inputs.

Examples:

- Amount cannot be negative.
- Transaction hour must be between 0 and 23.
- Device trust score must be between 0 and 100.
- Cardholder age must be within a valid range.
- Required fields must be present.
- Unexpected fields are rejected.

The API also handles:

- Invalid transaction data
- Missing fields
- Model inference errors
- Database errors
- Connection errors

---

# ⚙️ Setup Instructions

## Step 1 — Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-system.git
cd fraud-detection-system
```

## Step 2 — Create virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

## Step 4 — Start FastAPI

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Step 5 — Start Streamlit

Open another terminal:

```bash
streamlit run dashboard.py
```

---

# 🧪 Example Prediction

Input:

```json
{
  "amount": 850.50,
  "transaction_hour": 2,
  "merchant_category": "Electronics",
  "foreign_transaction": 1,
  "location_mismatch": 1,
  "device_trust_score": 25,
  "velocity_last_24h": 12,
  "cardholder_age": 29
}
```

The system processes the transaction:

```text
Transaction
     ↓
Validation
     ↓
Feature Engineering
     ↓
Preprocessing
     ↓
XGBoost
     ↓
Fraud Probability
     ↓
Risk Score
     ↓
Risk Level
```

---

# 📸 Screenshots

Add your actual screenshots to the repository using a folder such as:

```text
screenshots/
├── dashboard.png
├── prediction.png
├── fraud-transactions.png
├── risk-analysis.png
└── swagger-api.png
```

Then display them in this README:

### Dashboard

```markdown
![Fraud Detection Dashboard](screenshots/dashboard.png)
```

### Real-Time Prediction

```markdown
![Real-Time Fraud Prediction](screenshots/prediction.png)
```

### Fraud Transactions

```markdown
![Fraud Transactions](screenshots/fraud-transactions.png)
```

### API Documentation

```markdown
![FastAPI Swagger Documentation](screenshots/swagger-api.png)
```

> Replace the placeholder image paths with your actual screenshots before submitting the repository.

---

# ☁️ Deployment Details

The current project is structured for local execution with:

- FastAPI
- Streamlit
- SQLite

It can be extended for cloud deployment using:

```text
Docker
   ↓
FastAPI
   ↓
Cloud Platform
   ↓
PostgreSQL
   ↓
Streamlit
```

Possible deployment targets include cloud VM/container platforms and managed PostgreSQL services.

If cloud deployment is completed, add the live URLs here:

```text
API URL:
https://YOUR-API-URL

Dashboard URL:
https://YOUR-DASHBOARD-URL
```

If deployment is not available, the project can be evaluated locally using the setup instructions above.

---

# 🛠️ Technology Stack

## Programming

- Python
- NumPy
- Pandas

## Machine Learning

- Scikit-learn
- XGBoost
- Imbalanced-learn
- SMOTE
- Isolation Forest

## Visualization

- Matplotlib
- Seaborn
- Plotly

## Backend

- FastAPI
- REST API
- Pydantic

## Database

- SQLite
- SQLAlchemy
- PostgreSQL support

## Dashboard

- Streamlit

## Model Serialization

- Joblib

## Development

- Git
- GitHub

## Optional Deployment

- Docker
- Cloud deployment

---

# 📌 Key Features

✅ End-to-end fraud detection pipeline

✅ Data preprocessing and validation

✅ Exploratory data analysis

✅ Class imbalance handling

✅ SMOTE / over-sampling / under-sampling comparison

✅ Multiple ML model comparison

✅ XGBoost hyperparameter tuning

✅ Fraud probability prediction

✅ Risk scoring

✅ LOW / MEDIUM / HIGH risk classification

✅ Real-time FastAPI prediction

✅ Streamlit monitoring dashboard

✅ Prediction database

✅ Input validation

✅ Exception handling

✅ Model serialization

✅ Production-oriented project structure

---

# 🔮 Future Improvements

- Real-time transaction streaming using Kafka
- Advanced fraud feature engineering
- Model drift monitoring
- Data drift detection
- Automated model retraining
- MLflow experiment tracking
- Prometheus/Grafana monitoring
- Docker containerization
- Kubernetes deployment
- Cloud deployment
- Authentication and API authorization
- PostgreSQL production database
- Alerting for high-risk transactions

---

# 👨‍💻 Author

**Prem Mote**

Machine Learning / AI Engineer

GitHub: `https://github.com/moteprem4-web`

LinkedIn: `https://www.linkedin.com/in/prem-mote-898a99385`

---

# 📄 License

This project is created for educational, portfolio, and demonstration purposes.
<img width="1592" height="882" alt="image" src="https://github.com/user-attachments/assets/a960db20-8f39-4f65-9569-a5e077fc3dec" />
<img width="1422" height="687" alt="image" src="https://github.com/user-attachments/assets/bdff7d4f-3398-44c6-8e69-68b38562eb08" />
<img width="1527" height="677" alt="image" src="https://github.com/user-attachments/assets/63591363-824d-4459-9641-98157182f82d" />
<img width="1896" height="952" alt="image" src="https://github.com/user-attachments/assets/a6e6df49-73f2-4b5c-8193-cf0ffca7057f" />




