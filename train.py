import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for GitHub Actions
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate
from evidently import Report
from evidently.metric_preset import ClassificationPreset
from evidently.metrics import ClassificationClassBalance, ClassificationQualityByClass

# ===== 1. Load dataset from CSV =====
# Expecting file at data/iris.csv with same format as sklearn iris dataset
df = pd.read_csv("data/iris.csv")

# Assume target column is named 'species'
y = df['species']
X = df.drop(columns=['species'])

# ===== 2. Add random binary 'location' attribute =====
rng = np.random.default_rng(seed=42)
X['location'] = rng.integers(0, 2, size=len(X))

# ===== 3. Train/test split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# ===== 4. Train classifier =====
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

# ===== 5. Fairness evaluation wrt 'location' =====
y_pred = clf.predict(X_test)
y_true_bin = (y_test == 'virginica')
y_pred_bin = (y_pred == 'virginica')

metrics = {
    'accuracy': accuracy_score,
    'selection_rate': selection_rate
}

mf = MetricFrame(metrics=metrics,
                 y_true=y_true_bin,
                 y_pred=y_pred_bin,
                 sensitive_features=X_test['location'])

print("=== Fairness metrics by location group ===")
print(mf.by_group)
print("\n=== Overall metrics ===")
print(mf.overall)

# ===== 6. SHAP explanations for Virginica class =====
explainer = shap.Explainer(clf.predict_proba, X_train)
shap_values = explainer(X_test)

classes = clf.classes_
virginica_index = list(classes).index('virginica')
sv_vir = shap_values.values[:, :, virginica_index]

shap.summary_plot(sv_vir, X_test, feature_names=X_test.columns, show=False)
plt.title("SHAP Summary for Virginica Prediction")
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.close()
print("SHAP summary plot saved to shap_summary.png")

# ===== 7. Evidently classification report =====
test_df = X_test.copy()
test_df['target'] = y_test
test_df['prediction'] = y_pred

evidently_report = Report(metrics=[
    ClassificationPreset(),
    ClassificationClassBalance(),
    ClassificationQualityByClass()
])

evidently_report.run(reference_data=X_train.assign(target=y_train),
                     current_data=test_df)

evidently_report.save_html("iris_classification_report.html")
print("Evidently report saved to iris_classification_report.html")

