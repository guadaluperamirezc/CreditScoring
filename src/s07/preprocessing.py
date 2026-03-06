"""
preprocessing.py
────────────────────────────────────────────────────────────────────────────────
Preprocessing pipeline for the Credit Scoring dataset.

Context
───────
The pipeline is designed to be called from a main notebook and returns
a CLEAN, LEAKAGE-FREE feature matrix ready for modelling.

Data-leakage policy
───────────────────
The following variables are DROPPED from the feature set because they are
either post-event (realised AFTER the default decision) or are deterministic
derivations of the target variable (Default = 1):

    - DPD_30, DPD_60, DPD_90           → realised delinquency counts
    - Worst_Current_Status              → derived directly from DPDs
    - Months_Since_Most_Recent_
      Delinquency                       → post-event timing variable

These variables should only be used in descriptive/EDA sections,
NEVER as model inputs.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd


# ── Encoding maps ──
EDUCATION_MAP = {
    'High School' : 0,
    'Graduate'    : 1,
    'Postgraduate': 2,
}

MARITAL_MAP = {
    'Single'  : 0,
    'Married' : 1,
    'Divorced': 2,
    'Widowed' : 3,
}

# ── Monetary columns that benefit from log-compression ──
LOG_COLS = [
    'Income_INR',
    'Loan_Amount',
    'Savings_Account_Balance',
    'Checking_Account_Balance',
    'Total_Credit_Limit',
    'Total_Current_Balance',
    'Max_Credit_Exposure',
]

# ── Variables that leak the target — NEVER used as model features ──
LEAKAGE_COLS = [
    'DPD_30',
    'DPD_60',
    'DPD_90',
    'Worst_Current_Status',
    'Months_Since_Most_Recent_Delinquency',
]


class Preprocessing:
    """
    End-to-end preprocessing pipeline for the credit-scoring dataset.

    Parameters
    ----------
    raw_data_dir    : str  – folder that contains the raw CSV
    output_data_dir : str  – folder where the preprocessed CSV will be saved
    raw_filename    : str  – name of the raw CSV file
    output_filename : str  – name of the output CSV file
    drop_leakage    : bool – if True (default), leakage columns are removed
                            from the feature matrix (recommended for modelling)
    """

    def __init__(
        self,
        raw_data_dir: str,
        output_data_dir: str,
        raw_filename: str   = "credit_data.csv",
        output_filename: str = "preprocessed_credit_data.csv",
        drop_leakage: bool  = True,
    ):
        self.raw_data_dir    = raw_data_dir
        self.output_data_dir = output_data_dir
        self.raw_filename    = raw_filename
        self.output_filename = output_filename
        self.drop_leakage    = drop_leakage
        self.data            = None

    # ──────────────────────────────────────────────────────────────────────────
    # 1. LOAD
    # ──────────────────────────────────────────────────────────────────────────
    def load_data(self) -> None:
        """Read raw CSV from disk."""
        path = os.path.join(self.raw_data_dir, self.raw_filename)
        self.output_path = os.path.join(self.output_data_dir, self.output_filename)
        self.data = pd.read_csv(path)
        print(f"[load]  {len(self.data):,} rows × {self.data.shape[1]} cols loaded from '{path}'")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. STANDARDISE COLUMN NAMES
    # ──────────────────────────────────────────────────────────────────────────
    def _standardise_columns(self) -> None:
        """Strip, lower and normalise column names."""
        self.data.columns = (
            self.data.columns
            .str.strip()
            .str.replace(r'[\s\-]+', '_', regex=True)
            .str.lower()
        )
        print("[clean] Column names standardised.")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. DROP LEAKAGE COLUMNS
    # ──────────────────────────────────────────────────────────────────────────
    def _drop_leakage(self) -> None:
        """
        Remove post-event / target-derived variables.

        DATA LEAKAGE WARNING
        DPD_30/60/90, Worst_Current_Status and
        Months_Since_Most_Recent_Delinquency are realised AFTER the credit
        decision and must NOT be used as model inputs.  They are retained in
        the raw file for EDA and descriptive reporting only.
        """
        if self.drop_leakage:
            cols_present = [c for c in LEAKAGE_COLS if c in self.data.columns]
            self.data.drop(columns=cols_present, inplace=True)
            print(f"[leak]  Dropped leakage columns: {cols_present}")
        else:
            print("[leak]  Leakage columns kept (drop_leakage=False).")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. DROP ID & DATE (not informative for modelling)
    # ──────────────────────────────────────────────────────────────────────────
    def _drop_non_features(self) -> None:
        """Remove identifier and date columns."""
        self.data['Application_Date']  = pd.to_datetime(self.data['Application_Date'], errors='coerce')
        self.data['month'] = self.data['Application_Date'].dt.to_period('M')
        to_drop = ['Customer_ID', 'Application_Date']
        present = [c for c in to_drop if c in self.data.columns]
        self.data.drop(columns=present, inplace=True)
        print(f"[create] Created 'month' column from 'Application_Date'.")
        print(f"[clean] Dropped non-feature cols: {present}")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. DATA TYPES
    # ──────────────────────────────────────────────────────────────────────────
    def _fix_dtypes(self) -> None:
        """Ensure numeric columns have correct dtypes."""
        int_cols = [
            'Age', 'Employment_Years', 'Credit_History_Length',
            'Outstanding_Loans', 'Loan_Tenure_Months',
            'No_of_Open_Accounts', 'No_of_Closed_Accounts',
            'No_of_Inquiries_6M', 'No_of_Inquiries_12M',
            'Oldest_Trade_Open_Months', 'Newest_Trade_Open_Months',
            'Default',
        ]
        for col in int_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype(int)
        print("[dtype] Integer dtypes enforced.")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. DUPLICATES
    # ──────────────────────────────────────────────────────────────────────────
    def _drop_duplicates(self) -> None:
        before = len(self.data)
        self.data.drop_duplicates(inplace=True)
        removed = before - len(self.data)
        print(f"[clean] Duplicates removed: {removed} (remaining: {len(self.data):,})")

    # ──────────────────────────────────────────────────────────────────────────
    # 7. MISSING VALUES
    # ──────────────────────────────────────────────────────────────────────────
    def _handle_missing(self) -> None:
        """
        Strategy:
          - Numeric  → median imputation (robust to skew)
          - Categorical → mode imputation
          - Columns with > 70 % missing → dropped entirely
        """
        # Drop columns with > 70 % nulls
        high_null = [
            c for c in self.data.columns
            if self.data[c].isnull().mean() > 0.70
        ]
        if high_null:
            self.data.drop(columns=high_null, inplace=True)
            print(f"[miss]  Dropped high-null cols (>70 %): {high_null}")

        # Impute remaining
        num_cols = self.data.select_dtypes(include='number').columns
        cat_cols = self.data.select_dtypes(include='object').columns

        for col in num_cols:
            if self.data[col].isnull().any():
                med = self.data[col].median()
                self.data[col].fillna(med, inplace=True)
                print(f"[miss]  '{col}' → median imputed ({med:.2f})")

        for col in cat_cols:
            if self.data[col].isnull().any():
                mode = self.data[col].mode()[0]
                self.data[col].fillna(mode, inplace=True)
                print(f"[miss]  '{col}' → mode imputed ('{mode}')")

    # ──────────────────────────────────────────────────────────────────────────
    # 8. ENCODE CATEGORICALS
    # ──────────────────────────────────────────────────────────────────────────
    def _encode_categoricals(self) -> None:
        """
        Ordinal encoding for Education_Level and Marital_Status.
        Ordinal is preferred over one-hot for tree-based models and
        scorecard/logistic models with WoE binning downstream.
        """
        if 'Education_Level' in self.data.columns:
            self.data['Education_Level'] = (
                self.data['Education_Level'].map(EDUCATION_MAP)
            )
            print("[enc]   Education_Level → ordinal {HS:0, Grad:1, PG:2}")

        if 'Marital_Status' in self.data.columns:
            self.data['Marital_Status'] = (
                self.data['Marital_Status'].map(MARITAL_MAP)
            )
            print("[enc]   Marital_Status  → ordinal {Single:0, Married:1, Divorced:2, Widowed:3}")


    # ──────────────────────────────────────────────────────────────────────────
    # 9. LOG TRANSFORMATION (monetary variables)
    # ──────────────────────────────────────────────────────────────────────────
    def _log_transform(self) -> None:
        """
        Apply log1p to right-skewed monetary variables.
        Original columns are KEPT (prefixed with 'raw_') so that the
        scorecard / WoE analysis can work on interpretable rupee amounts.
        """
        for col in LOG_COLS:
            if col in self.data.columns:
                self.data[f'log_{col}'] = np.log1p(self.data[col].clip(lower=0))
        print(f"[trf]   log1p applied → {['log_' + c for c in LOG_COLS if c in self.data.columns]}")


    # ──────────────────────────────────────────────────────────────────────────
    # 12. REORDER COLUMNS (target last)
    # ──────────────────────────────────────────────────────────────────────────
    def _reorder_columns(self) -> None:
        """Put 'Default' as the last column (convention for ML pipelines)."""
        if 'Default' in self.data.columns:
            cols = [c for c in self.data.columns if c != 'Default'] + ['Default']
            self.data = self.data[cols]

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN PIPELINE
    # ──────────────────────────────────────────────────────────────────────────
    def clean_data(self) -> None:
        """Execute all preprocessing steps in order."""
        self._drop_leakage()          # ← must happen BEFORE any imputation
        self._drop_non_features()
        self._fix_dtypes()
        self._drop_duplicates()
        self._handle_missing()
        self._encode_categoricals()
        self._log_transform()
        self._standardise_columns()
        self._reorder_columns()

        os.makedirs(self.output_data_dir, exist_ok=True)
        self.data.to_csv(self.output_path, index=False)
        print(f"\n✅  Preprocessed data saved → '{self.output_path}'")
        print(f"    Shape: {self.data.shape[0]:,} rows × {self.data.shape[1]} cols")
        print(f"    Default rate: {self.data['default'].mean():.2%}")

    def run_all(self) -> pd.DataFrame:
        """Load + clean in one call. Returns the preprocessed DataFrame."""
        self.load_data()
        self.clean_data()
        return self.data

    # ──────────────────────────────────────────────────────────────────────────
    # QUICK REPORT
    # ──────────────────────────────────────────────────────────────────────────
    def report(self) -> None:
        """Print a brief data-quality summary after preprocessing."""
        if self.data is None:
            print("Run run_all() first.")
            return
        print("\n── Post-Preprocessing Summary ──────────────────────────────────")
        print(f"  Shape          : {self.data.shape}")
        print(f"  Missing values : {self.data.isnull().sum().sum()}")
        print(f"  Default rate   : {self.data['default'].mean():.2%}")
        print(f"  Dtypes         :\n{self.data.dtypes.value_counts().to_string()}")
        print("────────────────────────────────────────────────────────────────")