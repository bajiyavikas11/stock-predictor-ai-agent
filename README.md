# 📈 AI-Powered Stock Market Prediction & Investment Strategist

This project combines deep learning (LSTM with dual attention), real-time sentiment analysis, and multi-agent LLMs (via Gemini + Agno) to generate stock predictions and intelligent investment reports.

---

## 🔍 Project Overview

We built a full-stack predictive system that mimics how an investor evaluates both:
- **Structured data** (price, volume, market cap)
- **Unstructured data** (financial news sentiment)

Key features:
- 30-day window LSTM model with **temporal + feature attention**
- Real-time news sentiment scoring via **FinBERT**
- Investment assistant dashboard using **Streamlit**
- Modular AI agents (Market Analyst, Strategist, Company Research, Team Lead) powered by **Gemini via Agno**

---

## 📊 Files Included

| File                            | Description |
|---------------------------------|-------------|
| `predictor.py`                  | Core LSTM model with sentiment fusion and training pipeline |
| `Streamlit.py`                  | Streamlit app that runs prediction and displays reports |
| `Stock Market Prediction EDA.ipynb` | Exploratory Data Analysis |
---

## ⚙️ How to Run

1. **Clone the repo**  
```bash
git clone https://github.com/your-username/Stock-Prediction-AI-Agent.git
cd Stock-Prediction-AI-Agent
