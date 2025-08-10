import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from fairlearn.metrics import MetricFrame, selection_rate
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
import warnings
warnings.filterwarnings('ignore')

def create_iris_dataset():
    """Create iris dataset if it doesn't exist"""
    if not os.path.exists("data"):
        os.makedirs("data")
    
    if not os.path.exists("data/iris.csv"):
        # Create a sample iris dataset
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        df['species'] = iris.target_names[iris.target]
        df.to_csv("data/iris.csv", index=False)
        print("Created iris.csv dataset")
    return pd.read_csv("data/iris.csv")

def main():
    # ========= STEP 1: Load/Create data =========
    try:
        df = pd.read_csv("data/iris.csv")
    except FileNotFoundError:
        print("iris.csv not found. Creating dataset...")
        df = create_iris_dataset()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Classes: {df['species'].unique()}")
    
    # Add random sensitive attribute "location"
    np.random.seed(42)
    df["location"] = np.random.choice([0, 1], size=len(df))
    
    # Features and labels
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[feature_cols]
    y = df["species"]
    sensitive_features = df["location"]
    
    # ========= STEP 2: Train/Test Split =========
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive_features, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # ========= STEP 3: Train Model =========
    clf = RandomForestClassifier(random_state=42, n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Print basic metrics
    print(f"\n=== Model Performance ===")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    # ========= STEP 4: Fairness Metrics =========
    try:
        # For multi-class, we need to define what "positive" outcome means
        # Let's use 'virginica' as the positive class for fairness analysis
        y_test_binary = (y_test == 'virginica').astype(int)
        y_pred_binary = (y_pred == 'virginica').astype(int)
        
        mf = MetricFrame(
            metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
            y_true=y_test_binary,
            y_pred=y_pred_binary,
            sensitive_features=s_test
        )
        print("\n=== Fairness metrics by location group (Virginica vs Others) ===")
        print(mf.by_group)
        print("\n=== Overall metrics ===")
        print(mf.overall)
        
        # Check for significant disparities
        by_group_acc = mf.by_group['accuracy']
        if len(by_group_acc) > 1:
            acc_disparity = by_group_acc.max() - by_group_acc.min()
            print(f"\nAccuracy disparity between groups: {acc_disparity:.3f}")
            if acc_disparity > 0.1:
                print("⚠️  Potential fairness concern detected!")
            else:
                print("✓ Fairness metrics look reasonable")
    
    except Exception as e:
        print(f"Warning: Fairness metrics calculation failed: {e}")
    
    # ========= STEP 5: SHAP Explainability =========
    print("\n=== SHAP Explainability Analysis ===")
    
    # Create artifacts directory
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
    
    try:
        explainer = shap.TreeExplainer(clf)
        # Use a smaller sample for SHAP to avoid memory issues
        shap_sample = X_test.iloc[:20] if len(X_test) > 20 else X_test
        shap_values = explainer.shap_values(shap_sample)
        
        # For multi-class, shap_values is a list of arrays (one per class)
        class_names = clf.classes_
        
        # Create SHAP plots for each class
        for i, class_name in enumerate(class_names):
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values[i], 
                shap_sample, 
                feature_names=feature_cols,
                show=False,
                title=f'SHAP Summary for {class_name}'
            )
            plt.tight_layout()
            plt.savefig(f"artifacts/shap_summary_{class_name}.png", 
                       dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✓ SHAP summary plot saved for {class_name}")
        
        # Overall feature importance using mean absolute SHAP values
        plt.figure(figsize=(10, 6))
        # Average SHAP values across all classes
        mean_shap = np.mean([np.abs(shap_values[i]).mean(0) for i in range(len(class_names))], axis=0)
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': mean_shap
        }).sort_values('importance', ascending=True)
        
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Mean |SHAP value|')
        plt.title('Feature Importance (Average across all classes)')
        plt.tight_layout()
        plt.savefig("artifacts/feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("✓ Feature importance plot saved")
        
    except Exception as e:
        print(f"Warning: SHAP analysis failed: {e}")
        print("This might be due to library version incompatibilities")
    
    # ========= STEP 6: Evidently Drift & Summary =========
    print("\n=== Data Drift Analysis ===")
    
    try:
        # Define schema for Evidently
        schema = DataDefinition(
            numerical_columns=feature_cols,
            categorical_columns=["species", "location"],
        )
        
        # Create new data with intentional drift
        new_data = df.copy()
        
        # Add some realistic drift
        drift_indices = np.random.choice(len(new_data), size=int(0.2 * len(new_data)), replace=False)
        new_data.loc[drift_indices, 'sepal_length'] += np.random.normal(2, 0.5, len(drift_indices))
        new_data.loc[drift_indices, 'petal_width'] += np.random.normal(1, 0.3, len(drift_indices))
        
        # Add some extreme outliers to simulate data quality issues
        outlier_indices = np.random.choice(len(new_data), size=5, replace=False)
        new_data.loc[outlier_indices, 'sepal_length'] = 15.0
        new_data.loc[outlier_indices, 'petal_length'] = 10.0
        
        print(f"Original data shape: {df.shape}")
        print(f"New data shape: {new_data.shape}")
        
        # Create Evidently datasets
        eval_orig_data = Dataset.from_pandas(df, data_definition=schema)
        eval_new_data = Dataset.from_pandas(new_data, data_definition=schema)
        
        # Generate drift report
        report = Report(
            metrics=[DataDriftPreset(), DataSummaryPreset()],
            include_tests=True
        )
        
        report.run(eval_new_data, eval_orig_data)
        report.save_html("artifacts/drift_report.html")
        print("✓ Drift report saved to artifacts/drift_report.html")
        
        # Basic drift summary
        drift_results = report.as_dict()
        print("\nDrift Summary:")
        try:
            for metric in drift_results.get('metrics', []):
                if metric.get('metric') == 'DataDriftTable':
                    drift_detected = metric.get('result', {}).get('number_of_drifted_columns', 0)
                    total_columns = metric.get('result', {}).get('number_of_columns', 0)
                    print(f"  - {drift_detected}/{total_columns} columns show drift")
                    break
        except Exception as e:
            print(f"  - Could not parse detailed drift results: {e}")
    
    except Exception as e:
        print(f"Warning: Drift analysis failed: {e}")
        print("This might be due to Evidently version issues")
    
    # ========= STEP 7: Generate Summary Report =========
    print("\n" + "="*60)
    print("ML PIPELINE EXECUTION SUMMARY")
    print("="*60)
    
    summary = {
        "Dataset": f"{df.shape[0]} samples, {len(feature_cols)} features, {len(df['species'].unique())} classes",
        "Model": f"RandomForest with {clf.n_estimators} trees",
        "Accuracy": f"{accuracy_score(y_test, y_pred):.3f}",
        "Artifacts Generated": []
    }
    
    # Check which artifacts were created
    artifact_files = [
        "shap_summary_setosa.png",
        "shap_summary_versicolor.png", 
        "shap_summary_virginica.png",
        "feature_importance.png",
        "drift_report.html"
    ]
    
    for file in artifact_files:
        if os.path.exists(f"artifacts/{file}"):
            summary["Artifacts Generated"].append(file)
    
    for key, value in summary.items():
        if key == "Artifacts Generated":
            print(f"{key}:")
            for artifact in value:
                print(f"  ✓ {artifact}")
        else:
            print(f"{key}: {value}")
    
    print("\n✓ ML Pipeline completed successfully!")
    print(f"Check the 'artifacts' folder for generated visualizations and reports.")

if __name__ == "__main__":
    main()
