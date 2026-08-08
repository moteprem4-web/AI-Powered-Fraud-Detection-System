from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


API_URL = st.sidebar.text_input(
    "FastAPI URL",
    "http://127.0.0.1:8000",
).rstrip("/")


st.set_page_config(
    page_title="AI Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ AI Fraud Detection Monitoring Dashboard")
st.caption("Real-time monitoring powered by the trained XGBoost fraud model.")


def get_json(endpoint: str, params=None):
    response = requests.get(
        f"{API_URL}{endpoint}",
        params=params,
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


# -----------------------------
# Model performance
# -----------------------------
try:
    metrics = get_json("/metrics")
except requests.RequestException as exc:
    st.error(f"Cannot connect to FastAPI: {exc}")
    st.info("Start the API first: uvicorn app.main:app --reload")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Transactions", metrics["total_transactions"])
c2.metric("Fraud Transactions", metrics["fraud_transactions"])
c3.metric("Fraud Rate", f'{metrics["fraud_rate"]:.2f}%')
c4.metric("Average Risk", f'{metrics["average_risk_score"]:.2f}')
c5.metric("Avg API Latency", f'{metrics["average_latency_ms"]:.2f} ms')

st.divider()

# -----------------------------
# Real-time prediction
# -----------------------------
st.header("🔍 Real-Time Fraud Prediction")
st.write("Enter normal transaction details. You do NOT need to enter V1/V2/V3 or model-generated fields.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=250.0,
            step=10.0,
        )
        transaction_hour = st.slider(
            "Transaction Hour",
            0, 23, 12,
            help="0 = midnight, 12 = noon, 23 = 11 PM",
        )
        merchant_category = st.text_input(
            "Merchant Category",
            value="Electronics",
            help="Example: Electronics, Grocery, Travel, Fuel, Restaurant",
        )
        foreign_transaction = st.selectbox(
            "Foreign Transaction?",
            ["No", "Yes"],
        )

    with col2:
        location_mismatch = st.selectbox(
            "Location Mismatch?",
            ["No", "Yes"],
        )
        device_trust_score = st.slider(
            "Device Trust Score",
            0, 100, 75,
            help="0 = very suspicious device, 100 = highly trusted device",
        )
        velocity_last_24h = st.number_input(
            "Transactions in Last 24 Hours",
            min_value=0,
            value=2,
            step=1,
        )
        cardholder_age = st.number_input(
            "Cardholder Age",
            min_value=18,
            max_value=120,
            value=30,
            step=1,
        )

    submitted = st.form_submit_button(
        "🚀 Predict Fraud Risk",
        use_container_width=True,
    )

if submitted:
    payload = {
        "amount": amount,
        "transaction_hour": transaction_hour,
        "merchant_category": merchant_category.strip(),
        "foreign_transaction": int(foreign_transaction == "Yes"),
        "location_mismatch": int(location_mismatch == "Yes"),
        "device_trust_score": device_trust_score,
        "velocity_last_24h": int(velocity_last_24h),
        "cardholder_age": int(cardholder_age),
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()

            if result["prediction"] == 1:
                st.error("🚨 FRAUD TRANSACTION DETECTED")
            else:
                st.success("✅ TRANSACTION APPEARS NORMAL")

            p1, p2, p3, p4 = st.columns(4)
            p1.metric(
                "Fraud Probability",
                f'{result["fraud_probability"]:.2%}',
            )
            p2.metric(
                "Risk Score",
                f'{result["risk_score"]:.2f}/100',
            )
            p3.metric(
                "Risk Level",
                result["risk_level"],
            )
            p4.metric(
                "API Latency",
                f'{result["latency_ms"]:.2f} ms',
            )
        else:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            st.error(f"Prediction failed: {detail}")

    except requests.RequestException as exc:
        st.error(f"FastAPI connection error: {exc}")


st.divider()

# -----------------------------
# Monitoring data
# -----------------------------
st.header("📊 Transaction Monitoring")

limit = st.slider("Transactions to display", 10, 1000, 100)

try:
    data = get_json("/transactions", params={"limit": limit})
    df = pd.DataFrame(data)
except requests.RequestException as exc:
    st.error(f"Unable to load transaction history: {exc}")
    df = pd.DataFrame()

if df.empty:
    st.info("No predictions have been stored yet. Make a prediction above.")
else:
    fraud_df = df[df["prediction"] == 1].copy()

    st.subheader("🚨 Fraud Transactions")
    if fraud_df.empty:
        st.success("No fraud transactions in the selected records.")
    else:
        st.dataframe(
            fraud_df[
                [
                    "transaction_id",
                    "amount",
                    "merchant_category",
                    "fraud_probability",
                    "risk_score",
                    "risk_level",
                    "created_at",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("⚠️ Risk Level Distribution")
    risk_counts = (
        df["risk_level"]
        .value_counts()
        .rename_axis("risk_level")
        .reset_index(name="count")
    )

    fig = px.bar(
        risk_counts,
        x="risk_level",
        y="count",
        color="risk_level",
        title="Transactions by Risk Level",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Transaction Risk Pattern")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    fig = px.scatter(
        df.sort_values("created_at"),
        x="created_at",
        y="risk_score",
        color="risk_level",
        hover_data=[
            "amount",
            "merchant_category",
            "fraud_probability",
            "prediction",
        ],
        title="Risk Score Over Time",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💳 Fraud Probability Distribution")
    fig = px.histogram(
        df,
        x="fraud_probability",
        nbins=20,
        title="Fraud Probability Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏪 Fraud Pattern by Merchant Category")
    category_df = (
        df.groupby("merchant_category", as_index=False)["prediction"]
        .mean()
        .rename(columns={"prediction": "fraud_rate"})
    )
    category_df["fraud_rate"] *= 100

    fig = px.bar(
        category_df.sort_values("fraud_rate", ascending=False),
        x="merchant_category",
        y="fraud_rate",
        title="Observed Fraud Rate by Merchant Category",
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Offline model performance
# -----------------------------
st.subheader("🤖 Trained Model Performance")

model_metrics = metrics.get("model_metrics", {})

if model_metrics:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Precision", f'{model_metrics.get("Precision", 0):.4f}')
    m2.metric("Recall", f'{model_metrics.get("Recall", 0):.4f}')
    m3.metric("F1", f'{model_metrics.get("F1", 0):.4f}')
    m4.metric("ROC-AUC", f'{model_metrics.get("ROC-AUC", 0):.4f}')
    m5.metric("PR-AUC", f'{model_metrics.get("PR-AUC", 0):.4f}')
