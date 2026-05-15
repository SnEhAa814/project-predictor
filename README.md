# 🎓 Student Placement Predictor

A Machine Learning project that predicts whether a student will be placed based on academic
and skill-based features. Built with **scikit-learn**, **Flask**, and **Python**.

---

## 📁 Project Structure

```
placement_project/
├── data/
│   └── placement_data.csv      ← Generated dataset (1000 students)
├── model/
│   └── placement_model.pkl     ← Saved trained model
├── plots/
│   └── model_insights.png      ← Feature importance, confusion matrix, ROC curve
├── generate_data.py            ← Creates the dataset
├── train_model.py              ← Trains & evaluates the ML model
├── app.py                      ← Flask web app
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate dataset
```bash
mkdir data
python generate_data.py
```

### Step 3 — Train the model
```bash
python train_model.py
```

### Step 4 — Launch the web app
```bash
python app.py
```
Open http://localhost:5000 in your browser.

---

## 🧠 Features Used

| Feature           | Description                              |
|-------------------|------------------------------------------|
| CGPA              | Academic performance (4.0–10.0)          |
| Internships       | Number of internships completed          |
| Projects          | Number of personal/academic projects     |
| Skills            | Count of technical skills known          |
| Backlogs          | Number of active backlogs                |
| Communication     | Self-rated communication score (1–5)     |
| Aptitude Score    | Mock aptitude test score (%)             |
| Branch            | Engineering branch (CS/IT/ECE/Mech/Civil)|

---

## 📊 Models Compared

- Logistic Regression
- **Random Forest ← best performer**
- Gradient Boosting

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-learn** — ML models, pipelines, evaluation
- **Pandas / NumPy** — Data processing
- **Matplotlib / Seaborn** — Visualizations
- **Flask** — Web interface
- **Pickle** — Model serialization

---


---

*Built by Sneha Saxena — Final Year B.Tech, Computer Engineering, Vishwakarma University*
