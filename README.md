# 🛒 E-commerce Product Research AI Agent

> Multi-Agent AI system that researches ANY product category and finds your best selling opportunity!

![LangGraph](https://img.shields.io/badge/LangGraph-0.2.56-blue)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.3-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)

---

## 🎯 What It Does

Enter a product category like **"Wireless Earbuds"** and 6 AI agents automatically:

1. **Amazon Research Agent** — Finds top products and brands
2. **Price Analysis Agent** — Compares pricing and finds sweet spots
3. **Review Agent** — Analyzes customer feedback and gaps
4. **Trend Agent** — Researches market size and growth
5. **Opportunity Agent** — Synthesizes everything into your action plan

---

## 🏗️ Architecture

```
User enters Product Category
          ↓
  🎯 Supervisor Agent
          ↓
  🔍 Amazon Research Agent  → Finds top products
          ↓
  💰 Price Analysis Agent   → Compares pricing
          ↓
  ⭐ Review Agent           → Analyzes reviews
          ↓
  📊 Trend Agent            → Market trends
          ↓
  🧠 Opportunity Agent      → Best products to sell
          ↓
  Complete Research Report!
```

---

## ⚙️ Setup

### Step 1: Install
```bash
pip install -r requirements.txt
```

### Step 2: Configure
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### Step 3: Run Backend
```bash
python app.py
```

### Step 4: Run Frontend
```bash
streamlit run frontend/streamlit_app.py
```

Open: http://localhost:8501

---

## 🔑 API Keys Needed

| Key | Required | Get From |
|-----|----------|---------|
| GROQ_API_KEY | ✅ Yes | console.groq.com |
| SERPER_API_KEY | ⚡ Optional | serper.dev |

---

## 📊 Output

- **Opportunity Score** (0-100)
- **Recommended Products** to sell
- **Price Analysis** with segments
- **Review Insights** and gaps
- **Market Trends** and forecast
- **Go-to-Market Strategy**
- **Action Plan** with phases
- **Final Verdict** (GO/NO-GO)

---

## 💡 Example Use Cases

- Finding profitable Amazon FBA products
- Dropshipping product research
- Private label product selection
- Market gap analysis
- Competitive pricing strategy
