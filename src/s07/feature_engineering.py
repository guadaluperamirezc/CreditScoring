"""
feature_engineering.py
────────────────────────────────────────────────────────────────────────────────
Feature Engineering pipeline for the Credit Scoring dataset.

Context
───────
Classic Logistic Regression scorecard pipeline.
This script is designed to be called AFTER preprocessing.py and covers:

    1.  Derived features (ratios / domain knowledge)
    2.  Train / Test split  →  Out-of-Time (OOT) split by 'month'
                               Test  = last 4 months of 2025  (Sep–Dec)
                               Train = everything before
    3.  Correlation analysis  → drop highly correlated features (|r| > 0.85)
    4.  Weight of Evidence (WoE) encoding  →  fit on TRAIN, transform both
    5.  Information Value (IV) summary     →  feature selection guide

    WoE / IV are fitted EXCLUSIVELY on the training set.
    Applying them to the full dataset before splitting would leak
    future target information into the feature space.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── IV thresholds (Siddiqi, 2006) ─────────────────────────────────────────────
IV_LABELS = {
    (0.00, 0.02): 'Useless',
    (0.02, 0.10): 'Weak',
    (0.10, 0.30): 'Medium',
    (0.30, 0.50): 'Strong',
    (0.50, 9.99): 'Suspicious (possible leakage)',
}

# ── OOT cutoff ────────────────────────────────────────────────────────────────
OOT_START = pd.Period('2025-09', freq='M')   # Sep-2025 onwards → TEST (OOT)

# ── Correlation threshold ─────────────────────────────────────────────────────
CORR_THRESHOLD = 0.90

# ── Binning config for WoE (number of quantile bins per variable) ─────────────
N_BINS = 10


def _iv_label(iv: float) -> str:
    for (lo, hi), label in IV_LABELS.items():
        if lo <= iv < hi:
            return label
    return 'Unknown'


class FeatureEngineering:
    """
    Feature engineering pipeline for the credit-scoring dataset.

    Parameters
    ----------
    input_dir       : str  – folder with preprocessed CSV
    output_dir      : str  – folder to save outputs
    input_filename  : str  – preprocessed CSV filename
    target          : str  – name of the binary target column
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        input_filename: str = 'preprocessed_credit_data.csv',
        target: str         = 'default',
    ):
        self.input_dir      = input_dir
        self.output_dir     = output_dir
        self.input_filename = input_filename
        self.target         = target
        self.data           = None
        self.train          = None
        self.test           = None
        self.iv_summary     = None
        self._woe_maps      = {}       # fitted WoE maps  →  {col: Series}
        self._dropped_corr  = []       # columns removed by correlation filter

    # ──────────────────────────────────────────────────────────────────────────
    # 1. LOAD
    # ──────────────────────────────────────────────────────────────────────────
    def load_data(self) -> None:
        path = os.path.join(self.input_dir, self.input_filename)
        self.data = pd.read_csv(path)

        # Restore Period column if present as string
        if 'month' in self.data.columns:
            self.data['month'] = self.data['month'].astype(str).apply(
                lambda x: pd.Period(x, freq='M')
            )
        print(f"[load]  {len(self.data):,} rows × {self.data.shape[1]} cols loaded.")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. DERIVED FEATURES  (domain-driven ratios)
    # ──────────────────────────────────────────────────────────────────────────
    def _derived_features(self) -> None:
        """
        Ratios and interactions grounded in credit-risk theory.
        All divisions use eps to avoid ZeroDivisionError.
        """
        eps = 1e-6
        df  = self.data

        # ── Capacity ratios ───────────────────────────────────────────────────
        df['debt_to_income'] = (
            df['loan_amount'] / (df['income_inr'] + eps)
        ).round(4)

        df['balance_to_income'] = (
            df['total_current_balance'] / (df['income_inr'] + eps)
        ).round(4)

        df['savings_to_loan'] = (
            df['savings_account_balance'] / (df['loan_amount'] + eps)
        ).round(4)

        df['monthly_installment_proxy'] = (
            df['loan_amount'] / (df['loan_tenure_months'] + eps)
        ).round(2)

        df['installment_to_income'] = (
            df['monthly_installment_proxy'] / (df['income_inr'] / 12 + eps)
        ).round(4)

        # ── Credit behaviour ──────────────────────────────────────────────────
        df['inquiry_acceleration'] = (
            df['no_of_inquiries_6m'] / (df['no_of_inquiries_12m'] + eps)
        ).round(4)

        df['utilization_per_open_account'] = (
            df['credit_utilization_ratio'] / (df['no_of_open_accounts'] + eps)
        ).round(4)

        df['total_accounts'] = df['no_of_open_accounts'] + df['no_of_closed_accounts']

        df['closed_account_ratio'] = (
            df['no_of_closed_accounts'] / (df['total_accounts'] + eps)
        ).round(4)

        # ── Maturity / seniority ──────────────────────────────────────────────
        df['credit_hist_months'] = df['credit_history_length'] * 12

        df['account_maturity'] = (
            df['oldest_trade_open_months'] / (df['credit_hist_months'] + eps)
        ).round(4)

        df['trade_gap'] = (
            df['oldest_trade_open_months'] - df['newest_trade_open_months']
        )

        print(
            "[feat]  Derived features created:\n"
            "        debt_to_income, balance_to_income, savings_to_loan,\n"
            "        monthly_installment_proxy, installment_to_income,\n"
            "        inquiry_acceleration, utilization_per_open_account,\n"
            "        total_accounts, closed_account_ratio,\n"
            "        credit_hist_months, account_maturity, trade_gap"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 3. TRAIN / TEST SPLIT  —  Out-of-Time (OOT)
    # ──────────────────────────────────────────────────────────────────────────
    def _split_oot(self) -> None:
        """
        Temporal split: last 4 months of 2025 (Sep–Dec) → OOT test set.
        Everything before → training set.

        This replicates the regulatory requirement of validating the model
        on a hold-out period that was NOT seen during development.
        """
        if 'month' not in self.data.columns:
            raise ValueError(
                "'month' column not found. "
                "Ensure preprocessing.py created it from Application_Date."
            )

        oot_mask       = self.data['month'] >= OOT_START
        self.train     = self.data[~oot_mask].copy().reset_index(drop=True)
        self.test      = self.data[oot_mask].copy().reset_index(drop=True)

        print(
            f"[split] OOT split  →  "
            f"Train: {len(self.train):,} rows "
            f"({self.train[self.target].mean():.2%} default)  |  "
            f"Test (OOT ≥ {OOT_START}): {len(self.test):,} rows "
            f"({self.test[self.target].mean():.2%} default)"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 4. CORRELATION FILTER  (fit on train only)
    # ──────────────────────────────────────────────────────────────────────────
    def _correlation_filter(self) -> None:
        """
        Remove one of each highly correlated feature pair (|r| > threshold).
        Decision rule: keep the variable with higher absolute correlation
        with the target; drop the other.

        Computed on TRAIN set only.
        """
        feature_cols = [
            c for c in self.train.columns
            if c not in [self.target, 'month']
            and self.train[c].dtype in ['float64', 'int64']
        ]

        corr_matrix = self.train[feature_cols].corr().abs()
        upper       = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        target_corr = self.train[feature_cols].corrwith(
            self.train[self.target]
        ).abs()

        to_drop = set()
        for col in upper.columns:
            high_corr_peers = upper.index[upper[col] > CORR_THRESHOLD].tolist()
            for peer in high_corr_peers:
                # Drop whichever has lower correlation with target
                drop_candidate = col if target_corr.get(col, 0) < target_corr.get(peer, 0) else peer
                to_drop.add(drop_candidate)

        self._dropped_corr = list(to_drop)

        if self._dropped_corr:
            self.train.drop(columns=self._dropped_corr, inplace=True, errors='ignore')
            self.test.drop(columns=self._dropped_corr,  inplace=True, errors='ignore')
            print(f"[corr]  Dropped {len(self._dropped_corr)} highly correlated features "
                  f"(|r| > {CORR_THRESHOLD}): {self._dropped_corr}")
        else:
            print(f"[corr]  No features dropped at threshold |r| > {CORR_THRESHOLD}.")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. WoE ENCODING  (fit on train, transform both)
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_woe_iv(self) -> None:
            """
            Quantile-bin each numeric feature, then compute WoE and IV.

            Credit scoring convention:
            - Event     (Bad)  = 1
            - Non-event (Good) = 0
            - WoE(bin) = ln( Distribution_Good / Distribution_Bad )
            - IV        = Σ ( Distribution_Bad - Distribution_Good ) × WoE

            Fit on TRAIN only → apply to both TRAIN and TEST.
            """
            feature_cols = [
                c for c in self.train.columns
                if c not in [self.target, 'month']
                and self.train[c].dtype in ['float64', 'int64']
            ]

            iv_records = []
            if not hasattr(self, '_woe_maps'):
                self._woe_maps = {}

            for col in feature_cols:
                try:
                    # ── Bin on TRAIN ──────────────────────────────────────────────
                    train_binned, bin_edges = pd.qcut(
                        self.train[col],
                        q=N_BINS,
                        duplicates='drop',
                        retbins=True,
                    )

                    # ── WoE table from TRAIN ──────────────────────────────────────
                    tmp = pd.DataFrame({
                        'bin'   : train_binned,
                        'target': self.train[self.target].values,
                    })

                    stats = (
                        tmp.groupby('bin')['target']
                        .agg(events='sum', total='count')                          # events = Bad = 1
                        .assign(non_events=lambda x: x['total'] - x['events'])    # Good  = 0
                    )

                    total_events     = stats['events'].sum()
                    total_non_events = stats['non_events'].sum()

                    stats['dist_bad']  = (stats['events']     / (total_events     + 1e-9)).clip(lower=1e-9)
                    stats['dist_good'] = (stats['non_events'] / (total_non_events + 1e-9)).clip(lower=1e-9)

                    # WoE = ln(Good% / Bad%)  — positive WoE → more goods than bads in bin
                    stats['woe']   = np.log(stats['dist_good'] / stats['dist_bad'])
                    stats['iv_bin'] = (stats['dist_good'] - stats['dist_bad']) * stats['woe']

                    iv_total = float(stats['iv_bin'].sum())
                    woe_map  = stats['woe'].to_dict()   # {Interval → woe_value}

                    # ── Store fitted map ──────────────────────────────────────────
                    self._woe_maps[col] = {'bin_edges': bin_edges, 'woe_map': woe_map}

                    # ── Apply to TRAIN ────────────────────────────────────────────
                    # Convert to object first to avoid Categorical setitem error
                    self.train[f'woe_{col}'] = (
                        train_binned.astype(object).map(woe_map).fillna(0).astype(float)
                    )

                    # ── Apply to TEST (same bin edges) ────────────────────────────
                    test_binned = pd.cut(
                        self.test[col], bins=bin_edges, include_lowest=True
                    )
                    self.test[f'woe_{col}'] = (
                        test_binned.astype(object).map(woe_map).fillna(0).astype(float)
                    )

                    iv_records.append({
                        'Feature' : col,
                        'IV'      : round(iv_total, 4),
                        'Strength': _iv_label(iv_total),
                    })

                except Exception as e:
                    print(f"[woe]   Skipped '{col}': {e}")

            # ── IV summary — built ONCE after the loop ────────────────────────────
            self.iv_summary = (
                pd.DataFrame(iv_records)
                .sort_values('IV', ascending=False)
                .reset_index(drop=True)
            )

            print(f"[woe]   WoE encoding done. IV computed for {len(iv_records)} features.")
            print(f"\n── IV Summary ───────────────────────────────────────────────────")
            print(self.iv_summary.to_string(index=False))
            print(f"─────────────────────────────────────────────────────────────────")

            # ── Drop WoE columns that don't pass the IV threshold ─────────────────
            IV_MIN = 0.02   # weak (Siddiqi, 2006)
            keep_features = self.iv_summary.loc[self.iv_summary['IV'] >= IV_MIN, 'Feature'].tolist()
            woe_keep = {f'woe_{c}' for c in keep_features}
            woe_drop = [c for c in self.train.columns if c.startswith('woe_') and c not in woe_keep]

            if woe_drop:
                self.train.drop(columns=woe_drop, inplace=True, errors='ignore')
                self.test.drop(columns=woe_drop,  inplace=True, errors='ignore')

            print(f"\n[iv] Threshold IV ≥ {IV_MIN} → kept {len(keep_features)} features, dropped {len(woe_drop)}.")
            if keep_features:
                print(f"[iv] Selected: {[f'woe_{c}' for c in keep_features]}")
            else:
                print(f"[iv] No features passed IV ≥ {IV_MIN}. "
                    f"Consider lowering IV_MIN to 0.02 for synthetic data.")
            
    # ──────────────────────────────────────────────────────────────────────────
    # 6. SAVE
    # ──────────────────────────────────────────────────────────────────────────
    def _save(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        train_path  = os.path.join(self.output_dir, 'train_fe.csv')
        test_path   = os.path.join(self.output_dir, 'test_fe.csv')
        iv_path     = os.path.join(self.output_dir, 'iv_summary.csv')

        self.train.to_csv(train_path, index=False)
        self.test.to_csv(test_path,   index=False)
        self.iv_summary.to_csv(iv_path, index=False)

        print(
            f"\n✅  Outputs saved:\n"
            f"    Train  → '{train_path}'  ({len(self.train):,} rows × {self.train.shape[1]} cols)\n"
            f"    Test   → '{test_path}'   ({len(self.test):,} rows × {self.test.shape[1]} cols)\n"
            f"    IV     → '{iv_path}'"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN PIPELINE
    # ──────────────────────────────────────────────────────────────────────────
    def run_all(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute full feature engineering pipeline.
        Returns (train, test) DataFrames.
        """
        self.load_data()
        self._derived_features()
        self._split_oot()           # ← split BEFORE fitting WoE
        self._correlation_filter()  # ← fitted on train only
        self._compute_woe_iv()      # ← fitted on train, applied to both
        self._save()
        return self.train, self.test

    # ──────────────────────────────────────────────────────────────────────────
    # REPORTS
    # ──────────────────────────────────────────────────────────────────────────
    def report_iv(self, min_iv: float = 0.02) -> pd.DataFrame:
        """
        Print the IV table and return features above min_iv threshold.

        IV interpretation (Siddiqi, 2006):
          < 0.02  → Useless
          0.02 – 0.10  → Weak
          0.10 – 0.30  → Medium
          0.30 – 0.50  → Strong
          > 0.50  → Suspicious (check for leakage)
        """
        if self.iv_summary is None:
            print("Run run_all() first.")
            return pd.DataFrame()

        print("\n── IV Summary ───────────────────────────────────────────────────")
        print(self.iv_summary.to_string(index=False))
        print("─────────────────────────────────────────────────────────────────")

        selected = self.iv_summary[self.iv_summary['IV'] >= min_iv]['Feature'].tolist()
        woe_selected = [f'woe_{f}' for f in selected]
        print(f"\n[iv]    Features with IV ≥ {min_iv}: {len(selected)}")
        print(f"        → Use these WoE columns in your logistic regression:\n"
              f"          {woe_selected}")
        return self.iv_summary[self.iv_summary['IV'] >= min_iv]

    def report_split(self) -> None:
        """Show temporal distribution of train/test sets by month."""
        if self.train is None:
            print("Run run_all() first.")
            return

        print("\n── Temporal Split Distribution ─────────────────────────────────")

        for label, df in [('TRAIN', self.train), ('TEST (OOT)', self.test)]:
            summary = (
                df.groupby('month')[self.target]
                .agg(n='count', defaults='sum')
                .assign(default_rate=lambda x: (x['defaults'] / x['n']).round(4))
            )
            print(f"\n  {label}:")
            print(summary.to_string())

        print("─────────────────────────────────────────────────────────────────")