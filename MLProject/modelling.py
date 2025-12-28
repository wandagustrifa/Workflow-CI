import os
import json
import shutil
import argparse
import tempfile
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import 
import dagshub


# =========================
# TRACKING DAGSHUB
# =========================
dagshub.init(repo_owner="wandagustrifa", repo_name="diabetes-mlops-project", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/wandagustrifa/diabetes-mlops-project.mlflow")

def tune_and_train_model(input_filepath: str):
    print(f"Memuat data dari: {input_filepath}")
    df = pd.read_csv(input_filepath)

    if "Diagnosis" not in df.columns:
        raise ValueError("Kolom target 'Diagnosis' tidak ditemukan di dataset.")

    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    input_example = X_train.sample(n=5, random_state=42)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"LogisticRegression_Tuning_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        mlflow.sklearn.autolog(disable=True)

        pipeline = Pipeline([
            ("lr", LogisticRegression())
        ])

        param_grid = {
            "lr__C": [0.001, 0.01, 0.1, 1, 10],
            "lr__solver": ["liblinear", "lbfgs", "saga"],
            "lr__max_iter": [1000],
        }

        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1
        )

        print("\nMemulai Hyperparameter Tuning dengan GridSearchCV...")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # =========================
        # METRICS
        # =========================
        accuracy = accuracy_score(y_test, y_pred)
        precision_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        recall_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        f1_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

        precision_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        roc_auc = roc_auc_score(y_test, y_proba)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) != 0 else 0

        print("\nMetrik Model Terbaik (pada Test Set):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision Class 0: {precision_0:.4f}")
        print(f"Recall Class 0: {recall_0:.4f}")
        print(f"F1 Score Class 0: {f1_0:.4f}")
        print(f"Precision Class 1: {precision_1:.4f}")
        print(f"Recall Class 1: {recall_1:.4f}")
        print(f"F1 Score Class 1: {f1_1:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"False Positive Rate: {false_positive_rate:.4f}")

        # Log params & metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_class_0", precision_0)
        mlflow.log_metric("recall_class_0", recall_0)
        mlflow.log_metric("f1_score_class_0", f1_0)
        mlflow.log_metric("precision_class_1", precision_1)
        mlflow.log_metric("recall_class_1", recall_1)
        mlflow.log_metric("f1_score_class_1", f1_1)
        mlflow.log_metric("roc_auc_score", roc_auc)
        mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)
        mlflow.log_metric("specificity", specificity)
        mlflow.log_metric("false_positive_rate", false_positive_rate)

        # =========================
        # ARTIFACTS 
        # =========================
        temp_dir = "temp_mlflow_artifacts"
        os.makedirs(temp_dir, exist_ok=True)

        # Confusion Matrix PNG 
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Tidak Diabetes (0)", "Diabetes (1)"],
            yticklabels=["Tidak Diabetes (0)", "Diabetes (1)"]
        )
        plt.title("LogReg Confusion Matrix (Tuned)")
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        cm_path = os.path.join(temp_dir, "training_confussion_matrix.png")
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="") 

        # ROC Curve PNG 
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        roc_path = os.path.join(temp_dir, "roc_curve.png")
        plt.savefig(roc_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(roc_path, artifact_path="") 

        # metric_info.json 
        report_dict = {
            "accuracy": accuracy,
            "precision_class_0": precision_0,
            "recall_class_0": recall_0,
            "f1_score_class_0": f1_0,
            "precision_class_1": precision_1,
            "recall_class_1": recall_1,
            "f1_score_class_1": f1_1,
            "roc_auc_score": roc_auc,
            "specificity": specificity,
            "false_positive_rate": false_positive_rate,
            "best_cv_accuracy": grid_search.best_score_,
            "best_params": grid_search.best_params_,
        }
        metric_json_path = os.path.join(temp_dir, "metric_info.json")
        with open(metric_json_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=4)
        mlflow.log_artifact(metric_json_path, artifact_path="") 

        # estimator.html 
        estimator_html_path = os.path.join(temp_dir, "estimator.html")
        with open(estimator_html_path, "w", encoding="utf-8") as f:
            f.write("<html><body>")
            f.write("<h1>Best Estimator Parameters</h1>")
            f.write(f"<pre>{json.dumps(grid_search.best_params_, indent=4)}</pre>")
            f.write("<h1>Full Estimator</h1>")
            f.write(f"<pre>{str(best_model)}</pre>")
            f.write("</body></html>")
        mlflow.log_artifact(estimator_html_path, artifact_path="") 

        shutil.rmtree(temp_dir, ignore_errors=True)

        # =========================
        # MODEL 
        # =========================
        output_example = best_model.predict(input_example)
        signature = infer_signature(input_example, output_example)

        tmp_root = tempfile.mkdtemp()
        local_model_dir = os.path.join(tmp_root, "model")

        mlflow.sklearn.save_model(
            sk_model=best_model,
            path=local_model_dir,
            signature=signature,
            input_example=input_example,
            registered_model_name="Production_Diabetes_Model"
        )

        mlflow.log_artifacts(local_model_dir, artifact_path="model")
        run_id = mlflow.active_run().info.run_id
        shutil.rmtree(tmp_root, ignore_errors=True)

        print(f"\nMLflow Run ID: {run_id}")
        print(f"MLflow UI URL: {mlflow.get_tracking_uri()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, default="namadataset_preprocessing/preprocessed_diabetes_data.csv")
    args = parser.parse_args()
    
    tune_and_train_model(args.input_data)