# Holdpoint

**Flag the risk. Before the delay.**

Holdpoint is an AI-powered construction delay prediction tool built for 
project teams who want to act on risk before it becomes an overrun.

Upload your project schedule. Holdpoint analyses task-level features and 
predicts which activities are at high risk of delay — giving you time to 
intervene, not just report.

---

## The problem

9 in 10 major construction projects overrun their schedule. Most project 
management tools tell you a task is late after it is already late. 
Holdpoint flags the risk in advance.

---

## What it does

- Upload a project schedule (CSV or Excel)
- Holdpoint processes task features using a trained machine learning model
- Outputs a colour-coded risk dashboard showing which tasks need attention now

---

## Tech stack

- `pandas` / `openpyxl` — schedule data ingestion and processing
- `scikit-learn` — Random Forest delay prediction model
- `plotly` / `Streamlit` — interactive web interface and dashboard
- Deployed on Streamlit Cloud

---

## Status

🔨 Active development — Phase 1 (data pipeline + feature engineering) in progress.

---

## Roadmap

- [ ] Phase 1: Data ingestion pipeline and feature engineering
- [ ] Phase 2: ML model training and validation
- [ ] Phase 3: Streamlit web interface
- [ ] Phase 4: Public deployment and user testing
- [ ] Phase 5: User feedback and iteration

---

## About

Built by a civil engineer working at the intersection of AI and the built 
environment. Holdpoint is rooted in real construction practice — the feature 
engineering reflects domain knowledge that comes from working on live projects, 
not just reading about them.

---

## Contact

Moses Danso — [github.com/mosesdanso](https://github.com/mosesdanso)
