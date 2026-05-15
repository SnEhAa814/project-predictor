"""
Student Placement Predictor - Model Training
============================================
Trains, evaluates, and saves a Random Forest classifier.
Run: python train_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 55)
print("  STUDENT PLACEMENT PREDICTOR — MODEL TRAINING")
print("=" * 55)

df = pd.read_csv('data/placement_data.csv')
print(f"\n✓ Loaded {len(df)} records | Placement rate: {df['placed'].mean()*100:.1f}%")
print(df.head(3).to_string())

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
le = LabelEncoder()
df['branch_enc'] = le.fit_transform(df['branch'])

FEATURES = ['cgpa','internships','projects','skills','backlogs',
            'communication','aptitude_score','branch_enc']
TARGET   = 'placed'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✓ Train: {len(X_train)} | Test: {len(X_test)}")

# ── 3. Compare Multiple Models ─────────────────────────────────────────────────
print("\n--- Cross-Validation Scores (5-fold) ---")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

best_name, best_score = None, 0
for name, model in models.items():
    scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy')
    print(f"  {name:<25} Acc: {scores.mean():.3f} ± {scores.std():.3f}")
    if scores.mean() > best_score:
        best_score, best_name = scores.mean(), name

print(f"\n✓ Best model: {best_name} ({best_score:.3f})")

# ── 4. Train Best Model (Random Forest — usually wins) ────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    random_state=42
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  rf)
])

pipeline.fit(X_train, y_train)
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# ── 5. Evaluation ──────────────────────────────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n--- Final Model Evaluation ---")
print(f"  Accuracy : {acc*100:.2f}%")
print(f"  ROC-AUC  : {auc:.3f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Placed','Placed']))

# ── 6. Visualizations ──────────────────────────────────────────────────────────
os.makedirs('plots', exist_ok=True)
sns.set_theme(style='whitegrid', palette='muted')

# Feature Importance
importances = pipeline.named_steps['model'].feature_importances_
feat_df = pd.DataFrame({'Feature': FEATURES, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Student Placement Predictor — Model Insights', fontsize=14, fontweight='bold')

# Plot 1: Feature Importance
axes[0].barh(feat_df['Feature'], feat_df['Importance'], color='#4B0082', alpha=0.8)
axes[0].set_title('Feature Importance')
axes[0].set_xlabel('Importance Score')

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[1],
            xticklabels=['Not Placed','Placed'],
            yticklabels=['Not Placed','Placed'])
axes[1].set_title('Confusion Matrix')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

# Plot 3: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[2].plot(fpr, tpr, color='#4B0082', lw=2, label=f'AUC = {auc:.3f}')
axes[2].plot([0,1],[0,1],'--', color='gray', lw=1)
axes[2].set_title('ROC Curve')
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].legend()

plt.tight_layout()
plt.savefig('plots/model_insights.png', dpi=120, bbox_inches='tight')
print("\n✓ Plots saved to plots/model_insights.png")

# ── 7. Save Model ──────────────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
with open('model/placement_model.pkl', 'wb') as f:
    pickle.dump({'pipeline': pipeline, 'label_encoder': le, 'features': FEATURES}, f)

print("✓ Model saved to model/placement_model.pkl")
print("\n✅ Training complete! Now run: python app.py")
