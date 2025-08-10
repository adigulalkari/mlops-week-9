import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate
from scipy.stats import ks_2samp
import json
import warnings
warnings.filterwarnings('ignore')

# Create data and directories
os.makedirs("artifacts", exist_ok=True)
if not os.path.exists("data/iris.csv"):
    from sklearn.datasets import load_iris
    os.makedirs("data", exist_ok=True)
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    df['species'] = iris.target_names[iris.target]
    df.to_csv("data/iris.csv", index=False)
else:
    df = pd.read_csv("data/iris.csv")

# Prepare data
np.random.seed(42)
df["location"] = np.random.choice([0, 1], size=len(df))
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X, y = df[feature_cols], df["species"]

# Train model
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, df["location"], test_size=0.3, random_state=42, stratify=y)

clf = RandomForestClassifier(random_state=42, n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Fairness analysis
mf = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=(y_test == 'virginica').astype(int), 
    y_pred=(y_pred == 'virginica').astype(int), 
    sensitive_features=s_test
)
print("Fairness by group:")
print(mf.by_group)

# SHAP analysis
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X_test.values, feature_names=feature_cols, show=False, plot_type="bar")
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.savefig("artifacts/shap_plot.png", dpi=150, bbox_inches="tight")
plt.close()


# Drift detection
new_data = df.copy()
np.random.seed(123)
new_data.loc[np.random.choice(len(new_data), 30, replace=False), 'sepal_length'] += 2
drift_results = {col: {"drift_detected": bool(ks_2samp(df[col], new_data[col])[1] < 0.05)} 
                for col in feature_cols}
drifted = sum(r["drift_detected"] for r in drift_results.values())

with open("artifacts/drift_summary.json", 'w') as f:
    json.dump(drift_results, f, indent=2)

print(f"Drift: {drifted}/{len(feature_cols)} features drifted")
print(f"Generated {len(os.listdir('artifacts'))} artifacts")
