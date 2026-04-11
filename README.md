# 📊 Customer Segmentation & Churn Pattern Analytics
### European Banking | Unified Mentor Internship

> **Mentored by:** The European Central Bank (ECB)  
> **Domain:** Data Science  
> **Dataset:** 10,000 bank customers across France, Spain & Germany  
> **Status:** ✅ Complete — Notebooks + Dashboard + Reports

---

## 🧠 Project Overview

Customer churn is one of the most significant hidden costs in retail banking. When customers close their accounts, the bank loses lifetime value, incurs new acquisition costs, and faces revenue instability.

This project addresses that problem through **systematic, segmentation-driven churn analytics**. Using a dataset of 10,000 European bank customers, we:

- Identify which customer segments churn the most
- Quantify the financial risk from high-value customer churn
- Compare churn behavior across France, Spain, and Germany
- Deliver all findings through a live, interactive Streamlit dashboard

---

## 📁 Project Structure

```
C:\Projects\
│
├── 📂 app\
│   ├── main.py                          # Home page — KPI cards + overview charts
│   └── pages\
│       ├── 01_churn_summary.py          # Segment-wise churn rates (5 segment types)
│       ├── 02_geography_demographics.py # Country, age & gender analysis
│       ├── 03_highvalue_explorer.py     # Premium customer churn + revenue at risk
│       └── 04_engagement_tenure.py      # Activity levels + loyalty patterns
│
├── 📂 data\
│   ├── raw\
│   │   └── Churn_Modelling.csv          # Original Kaggle dataset (10,000 rows)
│   └── processed\
│       ├── churn_segmented.csv          # After EDA — 5 segment columns added
│       └── churn_final_scored.csv       # Final dataset — risk flags added (used by dashboard)
│
├── 📂 notebooks\
│   ├── 01_EDA.ipynb                     # Day 1–3: Exploratory Data Analysis (6 charts)
│   └── 02_churn_analysis.ipynb          # Day 4–7: Segmentation, KPIs & scoring (12 charts)
│
├── 📂 reports\
│   └── figures\                         # 18 chart PNGs (used in research paper)
│
├── requirements.txt                     # All Python dependencies
└── README.md                            # This file
```

---

## 📊 Dataset Description

| Column | Description |
|--------|-------------|
| `CreditScore` | Customer creditworthiness score |
| `Geography` | Country — France, Spain, or Germany |
| `Gender` | Male / Female |
| `Age` | Customer age in years |
| `Tenure` | Years with the bank |
| `Balance` | Account balance (€) |
| `NumOfProducts` | Number of bank products held |
| `HasCrCard` | Credit card ownership — 0 or 1 |
| `IsActiveMember` | Active engagement indicator — 0 or 1 |
| `EstimatedSalary` | Annual salary estimate (€) |
| `Exited` | **Target variable** — 1 = Churned, 0 = Retained |

> **Note:** `RowNumber`, `CustomerId`, and `Surname` are dropped during preprocessing — they carry no analytical value.

---

## 🎯 Key Performance Indicators (KPIs)

| # | KPI | Description | Where to Find |
|---|-----|-------------|----------------|
| 1 | **Overall Churn Rate** | % of all customers who exited | Home page — metric card |
| 2 | **Segment Churn Rate** | Churn % by age, credit, tenure, balance | Churn Summary page |
| 3 | **High-Value Churn Ratio** | Churn among top 25% balance customers | High-Value Explorer page |
| 4 | **Geographic Risk Index** | Regional churn exposure — 0 to 100 score | Geography & Demographics page |
| 5 | **Engagement Drop Indicator** | Inactive vs active member churn gap | Engagement & Tenure page |

---

## 🗂️ Customer Segments Created

| Segment Type | Values |
|--------------|--------|
| Age Segment | Under 30 / 30–45 / 46–60 / Over 60 |
| Credit Band | Low (< 580) / Medium (580–720) / High (> 720) |
| Tenure Group | New (0–2 yrs) / Mid-term (3–6 yrs) / Long-term (7+ yrs) |
| Balance Segment | Zero Balance / Low Balance / High Balance |
| Products Segment | 1 Product / 2 Products / 3+ Products |

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- pip

### Step 1 — Install all dependencies
```bash
cd C:\Projects
pip install -r requirements.txt
```

### Step 2 — Run Notebook 1 (EDA)
Open JupyterLab and run all cells in order:
```bash
jupyter lab
```
Then open `notebooks/01_EDA.ipynb` → Run All Cells  
This generates `data/processed/churn_segmented.csv` and 6 chart PNGs.

### Step 3 — Run Notebook 2 (Analysis)
Open `notebooks/02_churn_analysis.ipynb` → Run All Cells  
This generates `data/processed/churn_final_scored.csv` and 12 more chart PNGs.

### Step 4 — Launch the Streamlit Dashboard
```bash
cd C:\Projects
streamlit run app/main.py
```
The dashboard will open automatically at **http://localhost:8501**

> ⚠️ Always run `streamlit run` from `C:\Projects` — not from inside the `app\` folder.

---

## 📈 Dashboard Pages

| Page | File | What It Shows |
|------|------|---------------|
| 🏠 Home | `main.py` | 4 KPI cards, churn pie chart, country bar chart, age & activity charts |
| 📋 Churn Summary | `01_churn_summary.py` | Dropdown to explore any segment — churn rate + pie + data table |
| 🌍 Geography & Demographics | `02_geography_demographics.py` | Country × age × gender churn analysis |
| 💰 High-Value Explorer | `03_highvalue_explorer.py` | Balance slider, premium customer table, revenue at risk |
| 📉 Engagement & Tenure | `04_engagement_tenure.py` | Active vs inactive, tenure patterns, retention opportunities |

---

## 🔍 Key Findings

- **~20% overall churn rate** — 1 in 5 customers left the bank
- **Germany has the highest churn rate** — significantly above France and Spain
- **Customers aged 46–60 churn the most** across all geographies
- **Inactive members churn at nearly 2× the rate** of active members
- **High-balance customers churn more** — the revenue risk is concentrated at the top
- **3+ product customers churn dramatically more** — counterintuitive but consistent finding

---

## 📦 Dependencies

```
streamlit>=1.30.0
pandas>=2.0.0
plotly>=5.18.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🏛️ Project Information

| Field | Detail |
|-------|--------|
| Internship | Unified Mentor — Data Science Track |
| Mentor Organization | The European Central Bank (ECB) |
| Project Type | Segmentation + Churn Analytics |
| Deliverables | 2 Notebooks + Streamlit Dashboard + Research Paper |
| Dataset Source | Kaggle — Bank Customer Churn Modelling |

---

*Built as part of the Unified Mentor Data Science Internship Program.*
