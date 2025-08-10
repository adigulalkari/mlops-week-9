import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
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
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    iris_path = os.path.join(data_dir, "iris.csv")
    if not os.path.exists(iris_path):
        try:
            # Create a sample iris dataset
            from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
            df['species'] = iris.target_names[iris.target]
            df.to_csv(iris_path, index=False)
            print(f"✓ Created iris.csv dataset at {iris_path}")
        except Exception as e:
            print(f"❌ Failed to create iris dataset: {e}")
            sys.exit(1)
    
    return pd.read_csv(iris_path)

def ensure_artifacts_dir():
    """Ensure artifacts directory exists and is writable"""
    artifacts_dir = "artifacts"
    try:
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
        
        # Test write permissions
        test_file = os.path.join(artifacts_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        
        print(f"✓ Artifacts directory ready: {os.path.abspath(artifacts_dir)}")
        return artifacts_dir
    except Exception as e:
        print(f"❌ Failed to create/access artifacts directory: {e}")
        sys.exit(1)

def main():
    print("Starting ML Pipeline...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Ensure artifacts directory exists
    artifacts_dir = ensure_artifacts_dir()
    
    # ========= STEP 1: Load/Create data =========
    print("\n=== STEP 1: Loading Data ===")
    try:
        if os.path.exists("data/iris.csv"):
            df = pd.read_csv("data/iris.csv")
            print("✓ Loaded existing iris.csv")
        else:
            print("iris.csv not found. Creating dataset...")
            df = create_iris_dataset()
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        df = create_iris_dataset()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Classes: {df['species'].unique()}")
    print(f"Dataset head:\n{df.head()}")
    
    # Add random sensitive attribute "location"
    np.random.seed(42)
    df["location"] = np.random.choice([0, 1], size=len(df))
    
    # Features and labels
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[feature_cols]
    y = df["species"]
    sensitive_features = df["location"]
    
    # ========= STEP 2: Train/Test Split =========
    print("\n=== STEP 2: Data Splitting ===")
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive_features, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # ========= STEP 3: Train Model =========
    print("\n=== STEP 3: Model Training ===")
    clf = RandomForestClassifier(random_state=42, n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Print basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # ========= STEP 4: Fairness Metrics =========
    print("\n=== STEP 4: Fairness Analysis ===")
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
        print("Fairness metrics by location group (Virginica vs Others):")
        print(mf.by_group)
        print("\nOverall metrics:")
        print(mf.overall)
        
        # Check for significant disparities
        by_group_acc = mf.by_group['accuracy']
        if len(by_group_acc) > 1:
            acc_disparity = by_group_acc.max() - by_group_acc.min()
            print(f"Accuracy disparity between groups: {acc_disparity:.3f}")
            if acc_disparity > 0.1:
                print("⚠️  Potential fairness concern detected!")
            else:
                print("✓ Fairness metrics look reasonable")
        
        # Save fairness metrics to file
        fairness_report = {
            'by_group': mf.by_group.to_dict(),
            'overall': mf.overall.to_dict(),
            'disparity': float(acc_disparity) if len(by_group_acc) > 1 else 0.0
        }
        
        import json
        with open(os.path.join(artifacts_dir, 'fairness_metrics.json'), 'w') as f:
            json.dump(fairness_report, f, indent=2)
        print("✓ Fairness metrics saved")
    
    except Exception as e:
        print(f"⚠️  Fairness metrics calculation failed: {e}")
    
    # ========= STEP 5: SHAP Explainability =========
    print("\n=== STEP 5: SHAP Analysis ===")
    
    shap_success = False
    try:
        explainer = shap.TreeExplainer(clf)
        # Use a smaller sample for SHAP to avoid memory issues
        shap_sample = X_test.iloc[:20] if len(X_test) > 20 else X_test
        shap_values = explainer.shap_values(shap_sample)
        
        # For multi-class, shap_values is a list of arrays (one per class)
        class_names = clf.classes_
        print(f"Generating SHAP plots for classes: {class_names}")
        
        # Create SHAP plots for each class
        for i, class_name in enumerate(class_names):
            try:
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values[i], 
                    shap_sample, 
                    feature_names=feature_cols,
                    show=False
                )
                plt.title(f'SHAP Summary for {class_name}')
                plt.tight_layout()
                
                output_path = os.path.join(artifacts_dir, f"shap_summary_{class_name}.png")
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"✓ SHAP summary plot saved for {class_name}")
                shap_success = True
            except Exception as e:
                print(f"⚠️  Failed to create SHAP plot for {class_name}: {e}")
                plt.close()
        
        # Overall feature importance using mean absolute SHAP values
        try:
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
            
            output_path = os.path.join(artifacts_dir, "feature_importance.png")
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            print("✓ Feature importance plot saved")
            
            # Save feature importance as JSON too
            importance_data = feature_importance.to_dict('records')
            with open(os.path.join(artifacts_dir, 'feature_importance.json'), 'w') as f:
                json.dump(importance_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Failed to create feature importance plot: {e}")
            plt.close()
        
    except Exception as e:
        print(f"⚠️  SHAP analysis failed: {e}")
        print("This might be due to library version incompatibilities")
    
    # ========= STEP 6: Evidently Drift & Summary =========
    print("\n=== STEP 6: Data Drift Analysis ===")
    
    drift_success = False
    try:
        # Define schema for Evidently
        schema = DataDefinition(
            numerical_columns=feature_cols,
            categorical_columns=["species", "location"],
        )
        
        # Create new data with intentional drift
        new_data = df.copy()
        
        # Add some realistic drift
        np.random.seed(123)  # Different seed for reproducible but different drift
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
        
        drift_report_path = os.path.join(artifacts_dir, "drift_report.html")
        report.save_html(drift_report_path)
        print(f"✓ Drift report saved to {drift_report_path}")
        drift_success = True
        
        # Basic drift summary
        try:
            drift_results = report.as_dict()
            drift_summary = {"drift_detected": False, "details": "Could not parse drift results"}
            
            for metric in drift_results.get('metrics', []):
                if metric.get('metric') == 'DataDriftTable':
                    drift_detected = metric.get('result', {}).get('number_of_drifted_columns', 0)
                    total_columns = metric.get('result', {}).get('number_of_columns', 0)
                    drift_summary = {
                        "drift_detected": drift_detected > 0,
                        "drifted_columns": drift_detected,
                        "total_columns": total_columns,
                        "details": f"{drift_detected}/{total_columns} columns show drift"
                    }
                    print(f"Drift Summary: {drift_summary['details']}")
                    break
            
            # Save drift summary
            with open(os.path.join(artifacts_dir, 'drift_summary.json'), 'w') as f:
                json.dump(drift_summary, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Could not parse detailed drift results: {e}")
    
    except Exception as e:
        print(f"⚠️  Drift analysis failed: {e}")
        print("This might be due to Evidently version issues")
    
    # ========= STEP 7: Generate Summary Report =========
    print("\n" + "="*60)
    print("ML PIPELINE EXECUTION SUMMARY")
    print("="*60)
    
    # Create a comprehensive summary
    summary = {
        "dataset_info": {
            "samples": int(df.shape[0]),
            "features": len(feature_cols),
            "classes": len(df['species'].unique()),
            "class_names": df['species'].unique().tolist()
        },
        "model_info": {
            "type": "RandomForest",
            "n_estimators": int(clf.n_estimators),
            "accuracy": float(accuracy)
        },
        "pipeline_status": {
            "data_loading": True,
            "model_training": True,
            "fairness_analysis": 'fairness_metrics.json' in os.listdir(artifacts_dir) if os.path.exists(artifacts_dir) else False,
            "shap_analysis": shap_success,
            "drift_analysis": drift_success
        }
    }
    
    # Check which artifacts were created
    artifact_files = [
        "shap_summary_setosa.png",
        "shap_summary_versicolor.png", 
        "shap_summary_virginica.png",
        "feature_importance.png",
        "drift_report.html",
        "fairness_metrics.json",
        "feature_importance.json",
        "drift_summary.json"
    ]
    
    generated_artifacts = []
    for file in artifact_files:
        file_path = os.path.join(artifacts_dir, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            generated_artifacts.append({"name": file, "size_bytes": file_size})
            print(f"  ✓ {file} ({file_size} bytes)")
    
    summary["generated_artifacts"] = generated_artifacts
    
    # Save complete summary
    summary_path = os.path.join(artifacts_dir, 'pipeline_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\nDataset: {summary['dataset_info']['samples']} samples, {summary['dataset_info']['features']} features, {summary['dataset_info']['classes']} classes")
    print(f"Model: RandomForest with {summary['model_info']['n_estimators']} trees")
    print(f"Accuracy: {summary['model_info']['accuracy']:.3f}")
    print(f"\nGenerated Artifacts ({len(generated_artifacts)} files):")
    
    if generated_artifacts:
        for artifact in generated_artifacts:
            print(f"  ✓ {artifact['name']}")
    else:
        print("  ❌ No artifacts were generated!")
        sys.exit(1)
    
    print(f"\n✓ ML Pipeline completed successfully!")
    print(f"✓ All artifacts saved to: {os.path.abspath(artifacts_dir)}")
    
    # List directory contents for debugging
    print(f"\nArtifacts directory contents:")
    try:
        for item in os.listdir(artifacts_dir):
            item_path = os.path.join(artifacts_dir, item)
            size = os.path.getsize(item_path)
            print(f"  - {item} ({size} bytes)")
    except Exception as e:
        print(f"  ❌ Could not list directory: {e}")

if __name__ == "__main__":
    main()
