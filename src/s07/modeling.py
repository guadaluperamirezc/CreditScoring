"""
modeling.py
────────────────────────────────────────────────────────────────────────────────
Modelling pipeline for the Credit Scoring dataset.
Logistic Regression scorecard — Internal Validation ready.

Designed to be called from the main notebook after feature_engineering.py.

Usage
─────
    from modeling import Modeling

    mdl = Modeling(
        train_path = 'data/features/train_fe.csv',
        test_path  = 'data/features/test_fe.csv',
    )
    mdl.run_all()
────────────────────────────────────────────────────────────────────────────────
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics         import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, auc,
)

warnings.filterwarnings('ignore')


# ── Constants ─────────────────────────────────────────────────────────────────

TARGET      = 'default'
DROP_ALWAYS = ['month']          # non-feature columns to always exclude

# class_weight: use {0:1, 1:4} to reflect the 80/20 imbalance explicitly,
# or 'balanced' to let sklearn compute it automatically.
# Both are valid for a scorecard; {0:1,1:4} gives you more control.
CLASS_WEIGHT = {0: 1, 1: 4}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _select_features(df: pd.DataFrame, target: str) -> list[str]:
    """
    Keep only WoE-encoded columns as model inputs.
    Falls back to all numeric non-target columns if no woe_ columns exist.
    """
    woe_cols = [c for c in df.columns if c.startswith('woe_')]
    if woe_cols:
        return woe_cols
    # fallback: all numeric except target and drop list
    return [
        c for c in df.columns
        if c not in [target] + DROP_ALWAYS
        and df[c].dtype in ['float64', 'int64']
    ]


# def _find_best_threshold(y_true, y_proba) -> float:
#     """
#     Find the probability cut-off that maximises F1 on the training predictions.
#     Used to move away from the default 0.5 threshold given class imbalance.
#     """
#     precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
#     f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
#     best_idx  = np.argmax(f1_scores[:-1])   # last element has no threshold
#     return float(thresholds[best_idx])


def _find_best_threshold(y_true, y_proba, metric: str = "f1") -> float:
    """
    Selects the best threshold according to the chosen criterion:
    - 'f1': maximizes F1 (using the Precision-Recall curve)
    - 'recall': maximizes Recall (pure)
    - 'auc': since AUC/Gini do not depend on the threshold, returns the operational threshold that maximizes KS (TPR - FPR)

    y_true: array-like with labels {0,1} (1 = Bad/event)
    y_proba : array-like with predicted event probabilities
    metric: 'f1', 'recall', or 'auc'
    """
    metric = metric.lower().strip()

    # ---  F1 ---
    if metric == "f1":
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
        best_idx  = np.argmax(f1_scores[:-1])  # el último no tiene threshold
        return float(thresholds[best_idx])

    # ---  Recall ---
    elif metric == "recall":
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        best_idx = np.argmax(recalls[:-1])
        return float(thresholds[best_idx])

    # --- Caso "auc" → usar umbral KS máximo ---
    elif metric == "auc":
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        ks_values = tpr - fpr
        best_idx = np.argmax(ks_values)
        return float(thresholds[best_idx])

    else:
        raise ValueError("The metric has to be one of these: {'f1', 'recall', 'auc'}")

# ── Main class ────────────────────────────────────────────────────────────────

class Modeling:
    """
    Logistic Regression scorecard pipeline.

    Parameters
    ----------
    train_path  : str   – path to train_fe.csv
    test_path   : str   – path to test_fe.csv (OOT)
    output_dir  : str   – folder to save outputs
    target      : str   – name of the binary target column
    n_iter      : int   – RandomizedSearchCV iterations
    n_splits    : int   – StratifiedKFold splits
    """

    def __init__(
        self,
        train_path : str,
        test_path  : str,
        output_dir : str  = '/model/',
        figure_dir : str = '/figuras/',
        target     : str  = TARGET,
        n_iter     : int  = 10,
        n_splits   : int  = 5,
    ):
        self.train_path = train_path
        self.test_path  = test_path
        self.output_dir = output_dir
        self.figure_dir = figure_dir
        self.target     = target
        self.n_iter     = n_iter
        self.n_splits   = n_splits

        # populated during run_all()
        self.X_train    = None
        self.y_train    = None
        self.X_test     = None
        self.y_test     = None
        self.best_model = None
        self.threshold  = 0.5
        self.features   = []

    # ──────────────────────────────────────────────────────────────────────────
    # 1. LOAD & SPLIT X / y
    # ──────────────────────────────────────────────────────────────────────────
    def load_data(self) -> None:
        train = pd.read_csv(self.train_path)
        test  = pd.read_csv(self.test_path)

        train.columns = train.columns.str.lower().str.strip()
        test.columns  = test.columns.str.lower().str.strip()

        self.features = _select_features(train, self.target)

        self.train_month = train['month'].astype(str) if 'month' in train.columns else None
        self.test_month  = test['month'].astype(str)  if 'month' in test.columns  else None

        self.X_train = train[self.features]
        self.y_train = train[self.target]
        self.X_test  = test[self.features]
        self.y_test  = test[self.target]

        print(f"[load]  Train: {self.X_train.shape}  |  Test (OOT): {self.X_test.shape}")
        print(f"[load]  Features used ({len(self.features)}): {self.features}")
        print(f"[load]  Train default rate: {self.y_train.mean():.2%}  |  "
              f"Test default rate: {self.y_test.mean():.2%}")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. TRAIN — RandomizedSearchCV + StratifiedKFold
    # ──────────────────────────────────────────────────────────────────────────
    def train(self) -> None:
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  LogisticRegression(
                class_weight = CLASS_WEIGHT,
                random_state = 42,
                max_iter     = 1000,
            ))
        ])

        param_grid = {
            'model__C'      : [0.001, 0.01, 0.1, 1, 10],
            'model__penalty': ['l1', 'l2'],
            'model__solver' : ['liblinear', 'saga'],
        }

        search = RandomizedSearchCV(
            estimator            = pipeline,
            param_distributions  = param_grid,
            n_iter               = self.n_iter,
            scoring              = 'f1_macro',   # AUC is standard for PD models recall, roc_auc
            cv                   = cv,
            verbose              = 0,
            n_jobs               = -1,
            random_state         = 42,
            refit                = True,
        )

        print("\n[train] Running RandomizedSearchCV (Logistic Regression)...")
        search.fit(self.X_train, self.y_train)

        self.best_model = search.best_estimator_

        print(f"[train] Best params    : {search.best_params_}")
        print(f"[train] Best CV AUC    : {search.best_score_:.4f}")

        # ── Optimal threshold (from train predictions) ────────────────────────
        train_proba    = self.best_model.predict_proba(self.X_train)[:, 1]
        self.threshold = _find_best_threshold(self.y_train, train_proba, 'f1')
        print(f"[train] Optimal threshold (max F1 on train): {self.threshold:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. EVALUATE — on OOT test set
    # ──────────────────────────────────────────────────────────────────────────
    def evaluate(self) -> tuple[np.ndarray, np.ndarray]:
        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        y_pred  = (y_proba >= self.threshold).astype(int)

        print(f"\n── Test Set Metrics (OOT)  —  threshold = {self.threshold:.4f} ────────")
        print(f"  ROC AUC          : {roc_auc_score(self.y_test, y_proba):.4f}")
        print(f"  PR AUC           : {average_precision_score(self.y_test, y_proba):.4f}")
        print(f"  Recall           : {recall_score(self.y_test, y_pred):.4f}")
        print(f"  Precision        : {precision_score(self.y_test, y_pred):.4f}")
        print(f"  F1 Score         : {f1_score(self.y_test, y_pred):.4f}")
        print(f"\n{classification_report(self.y_test, y_pred, target_names=['Good (0)', 'Bad (1)'])}")

        return y_proba, y_pred

    # ──────────────────────────────────────────────────────────────────────────
    # 4. PLOTS
    # ──────────────────────────────────────────────────────────────────────────
    def plot_evaluation(self, y_proba: np.ndarray, y_pred: np.ndarray) -> None:
        precisions, recalls, _ = precision_recall_curve(self.y_test, y_proba)
        pr_auc = auc(recalls, precisions)
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        cm = confusion_matrix(self.y_test, y_pred)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Logistic Regression — OOT Evaluation', fontsize=13, fontweight='bold')

        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')

        # ROC Curve
        axes[1].plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}', color='steelblue')
        axes[1].plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        axes[1].set_title('ROC Curve')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].legend()

        # Precision-Recall Curve
        axes[2].plot(recalls, precisions, label=f'PR AUC = {pr_auc:.3f}', color='darkorange')
        axes[2].axhline(self.y_test.mean(), color='grey', linestyle='--',
                        linewidth=0.8, label=f'Baseline = {self.y_test.mean():.2%}')
        axes[2].set_title('Precision-Recall Curve')
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].legend()

        plt.tight_layout()

        os.makedirs(self.output_dir, exist_ok=True)
        fig_path = os.path.join(self.figure_dir, 'lr_evaluation.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"[plot]  Saved → '{fig_path}'")

    def plot_coefficients(self) -> None:
        """Bar chart of logistic regression coefficients (feature importance proxy)."""
        lr_step = self.best_model.named_steps['model']
        coefs   = pd.Series(lr_step.coef_[0], index=self.features).sort_values()

        fig, ax = plt.subplots(figsize=(8, max(4, len(coefs) * 0.35)))
        colors  = ['#d73027' if v > 0 else '#4575b4' for v in coefs]
        coefs.plot(kind='barh', ax=ax, color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title('Logistic Regression — Coefficients\n(Red = increases default risk)', fontsize=11)
        ax.set_xlabel('Coefficient value')
        plt.tight_layout()

        fig_path = os.path.join(self.figure_dir, 'lr_coefficients.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"[plot]  Saved → '{fig_path}'")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. SAVE PREDICTIONS
    # ──────────────────────────────────────────────────────────────────────────
    def save_predictions(self, y_proba: np.ndarray, y_pred: np.ndarray) -> None:
        test_df = pd.read_csv(self.test_path)
        test_df['score_proba'] = y_proba
        test_df['pred_label']  = y_pred

        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, 'test_predictions.csv')
        test_df.to_csv(out_path, index=False)
        print(f"[save]  Predictions saved → '{out_path}'")

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN PIPELINE
    # ──────────────────────────────────────────────────────────────────────────
    def run_all(self) -> None:
        self.load_data()
        self.train()
        y_proba, y_pred = self.evaluate()
        self.plot_evaluation(y_proba, y_pred)
        self.plot_coefficients()
        self.save_predictions(y_proba, y_pred)
        print("\n✅  Modelling pipeline complete.")