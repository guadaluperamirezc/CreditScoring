"""
generate_credit_data_v3.py
══════════════════════════════════════════════════════════════════════════════
Synthetic credit dataset with EMPIRICALLY-MOTIVATED relationships between
ALL features — raw AND derived — and the Default target.

Economic logic per variable (sign vs. default probability)
──────────────────────────────────────────────────────────
RAW FEATURES
  Income_INR                  (−) Higher income → stronger repayment capacity
  Education_Level             (−) Higher education → financial literacy, stability
  Employment_Years            (−) Longer tenure → job stability
  Age                         (−/+) U-shaped; very young and very old = more risk
  Marital_Status              (−) Married = slightly more stable
  Credit_History_Length       (−) Longer history → proven payer
  Credit_Utilization_Ratio    (+) High utilisation → financial stress
  No_of_Inquiries_12M         (+) Many inquiries → desperation / over-leverage
  Outstanding_Loans           (+) More obligations → capacity constrained
  Loan_Amount                 (+) Larger loan → higher obligation burden
  Loan_Tenure_Months          (+) Longer term → more time-at-risk (marginal)
  Savings_Account_Balance     (−) Liquid buffer → absorbs shocks
  Checking_Account_Balance    (−) Liquid buffer
  Total_Credit_Limit          (−) Higher limit → institution trust proxy
  Total_Current_Balance       (+) Higher outstanding → debt burden
  Max_Credit_Exposure         (+) Higher exposure → more systemic risk
  Oldest_Trade_Open_Months    (−) Old accounts → reliability
  Newest_Trade_Open_Months    (+) Very recent accounts → recency of credit need
  No_of_Open_Accounts         (+) Too many open accounts → juggling
  No_of_Closed_Accounts       (−) Closed accounts → settled debts, maturity

DERIVED FEATURES  (created by feature_engineering.py)
  debt_to_income              (+) High leverage → default risk ↑
  savings_to_loan             (−) More savings relative to loan → safer
  installment_to_income       (+) High payment burden relative to income → risk ↑
  inquiry_acceleration        (+) More inquiries in last 6M vs 12M → urgency ↑
  account_maturity            (−) Older accounts relative to history → stability ↑
  trade_gap                   (−) Wide gap between oldest and newest → stable history
  credit_hist_months          (−) Longer history (months version) → safer

DPD / delinquency flags   →  LEAKAGE, not in model, but used to construct target
══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from scipy.special import expit

np.random.seed(42)
N = 2000

# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────
def random_dates(start, end, n):
    s = pd.Timestamp(start).value // 10**9
    e = pd.Timestamp(end).value   // 10**9
    return pd.to_datetime(np.random.randint(s, e, n), unit='s').strftime('%Y-%m-%d')

def zscore(x):
    """Standardise array → mean=0, std=1 for logit construction."""
    return (x - x.mean()) / (x.std() + 1e-9)

eps = 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# 1. DEMOGRAPHICS
# ─────────────────────────────────────────────────────────────────────────────
ages = np.random.randint(21, 70, N)

edu = np.random.choice(
    ['High School', 'Graduate', 'Postgraduate'],
    N, p=[0.25, 0.50, 0.25]
)
edu_num = np.where(edu == 'High School', 0,
          np.where(edu == 'Graduate',    1, 2))  # 0/1/2

# Income: positively correlated with age + education
base_income = (ages - 20) * 3_000
edu_boost   = np.where(edu == 'High School', 0,
              np.where(edu == 'Graduate',    150_000, 300_000))
income = (base_income + edu_boost +
          np.random.normal(0, 60_000, N)).clip(100_000, 5_000_000).astype(int)

# Employment: correlated with age
employment_years = np.clip(
    (ages - 22) * np.random.uniform(0.5, 1.0, N) + np.random.normal(0, 2, N),
    0, 40
).astype(int)

marital = np.random.choice(
    ['Single', 'Married', 'Divorced', 'Widowed'],
    N, p=[0.35, 0.45, 0.15, 0.05]
)
# Married = slight buffer; divorced/widowed = mild stress
marital_risk = np.where(marital == 'Married',  -0.12,
               np.where(marital == 'Single',    0.00,
               np.where(marital == 'Divorced',  0.10,  0.08)))

# ─────────────────────────────────────────────────────────────────────────────
# 2. CREDIT HISTORY
# ─────────────────────────────────────────────────────────────────────────────
credit_hist_len = np.clip(
    employment_years * 0.8 + np.random.randint(1, 10, N), 1, 30
).astype(int)
credit_hist_months = credit_hist_len * 12

no_open   = np.random.randint(1, 8, N)
no_closed = np.random.randint(0, 10, N)

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAN DETAILS
# ─────────────────────────────────────────────────────────────────────────────
loan_factor = np.random.uniform(0.5, 5.0, N)
loan_amount = (income * loan_factor).clip(50_000, 10_000_000).astype(int)
loan_tenure = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240], N)
outstanding = np.random.randint(0, 5, N)

# ─────────────────────────────────────────────────────────────────────────────
# 4. BALANCES & CREDIT LIMITS
# ─────────────────────────────────────────────────────────────────────────────
# Savings: higher income → higher savings (but with noise)
savings  = (income * np.random.uniform(0.05, 2.0, N)).clip(0, 5_000_000).astype(int)
checking = (income * np.random.uniform(0.02, 0.5, N)).clip(0,   1_000_000).astype(int)

total_credit_limit    = (income * np.random.uniform(0.5, 3.0, N)).clip(50_000, 10_000_000).astype(int)
credit_util_ratio     = np.random.beta(2, 5, N).round(2)   # skewed low (realistic)
total_current_balance = (total_credit_limit * credit_util_ratio).astype(int)
max_credit_exp        = (total_credit_limit * np.random.uniform(1.0, 1.5, N)).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 5. INQUIRIES
# ─────────────────────────────────────────────────────────────────────────────
inq_6m  = np.random.poisson(1.2, N)
inq_12m = inq_6m + np.random.poisson(0.8, N)   # always >= inq_6m

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRADE AGES
# ─────────────────────────────────────────────────────────────────────────────
oldest_trade = credit_hist_months + np.random.randint(0, 12, N)
newest_trade = np.random.randint(1, oldest_trade + 1)

# ─────────────────────────────────────────────────────────────────────────────
# 7. DERIVED FEATURES  (mirror exactly what feature_engineering.py computes)
# ─────────────────────────────────────────────────────────────────────────────
debt_to_income         = loan_amount        / (income       + eps)
savings_to_loan        = savings            / (loan_amount  + eps)
installment_proxy      = loan_amount        / (loan_tenure  + eps)
installment_to_income  = installment_proxy  / (income / 12  + eps)
inquiry_acceleration   = inq_6m             / (inq_12m      + eps)
account_maturity       = oldest_trade       / (credit_hist_months + eps)
trade_gap              = oldest_trade - newest_trade

# ─────────────────────────────────────────────────────────────────────────────
# 8. DPD  (post-event — for target construction only, dropped in modelling)
# ─────────────────────────────────────────────────────────────────────────────
dpd_mask = np.random.random(N) < 0.20
dpd30    = np.where(dpd_mask, np.random.randint(1, 5, N), 0)
dpd60    = np.where(dpd_mask, np.random.randint(0, dpd30.clip(1) + 1), 0)
dpd90    = np.where(dpd_mask, np.random.randint(0, dpd60.clip(1) + 1), 0)

worst_status = np.where(
    dpd90 > 0, 'DPD90+',
    np.where(dpd60 > 0, 'DPD60',
    np.where(dpd30 > 0, 'DPD30', 'Current'))
)
months_since_delinq = np.where(dpd_mask, np.random.randint(1, 36, N), np.nan)

# ═════════════════════════════════════════════════════════════════════════════
# 9. DEFAULT LABEL — logistic scoring covering ALL features & derived features
# ═════════════════════════════════════════════════════════════════════════════
#
#  Each term: (coefficient) × zscore(variable)
#  Standardising first lets coefficients reflect relative importance directly.
#  Signs are set by the economic logic documented at the top of this file.
#
#  DPD flags are kept as strong signals because in reality they ARE the best
#  predictors — they will be dropped from the model (leakage policy) but they
#  anchor the target to a realistic distribution.
#
log_odds = (
    # ── Intercept (calibrates baseline default rate to ~22–27 %) ────────────
    -2.20

    # ── Income & capacity  (−) ───────────────────────────────────────────────
    - 0.75 * zscore(np.log1p(income))            # higher income → safer
    - 0.45 * zscore(edu_num)                     # higher education → safer
    - 0.40 * zscore(employment_years)            # job stability → safer
    - 0.30 * zscore(credit_hist_len)             # long history → safer
    + marital_risk                               # marital status (pre-scaled)

    # ── Age: U-shaped risk (young & old riskier) — quadratic term ───────────
    + 0.20 * zscore((ages - 45) ** 2)            # distance from peak-earning age

    # ── Leverage & burden  (+) ───────────────────────────────────────────────
    + 0.55 * zscore(debt_to_income)              # main leverage ratio
    + 0.50 * zscore(installment_to_income)       # payment-to-income burden
    + 0.35 * zscore(credit_util_ratio)           # utilisation pressure
    + 0.30 * zscore(total_current_balance)       # total outstanding
    + 0.25 * zscore(outstanding)                 # number of active loans
    + 0.20 * zscore(loan_amount)                 # raw loan size

    # ── Liquidity buffers  (−) ───────────────────────────────────────────────
    - 0.50 * zscore(savings_to_loan)             # savings cushion vs loan
    - 0.30 * zscore(np.log1p(savings))           # log savings (level)
    - 0.20 * zscore(np.log1p(checking))          # checking buffer
    - 0.20 * zscore(np.log1p(total_credit_limit))# institutional trust proxy

    # ── Inquiry & behavioural signals  (+) ──────────────────────────────────
    + 0.40 * zscore(inq_12m)                     # inquiry pressure (12M)
    + 0.25 * zscore(inquiry_acceleration)        # recent spike in inquiries
    + 0.20 * zscore(no_open)                     # too many open lines

    # ── Credit maturity & account history  (−) ──────────────────────────────
    - 0.30 * zscore(account_maturity)            # old accounts = reliability
    - 0.25 * zscore(trade_gap)                   # wide gap = stable history
    - 0.20 * zscore(credit_hist_months)          # long history (months)
    - 0.15 * zscore(oldest_trade)                # age of oldest trade line
    + 0.15 * zscore(newest_trade)                # very recent new account = risk
    - 0.10 * zscore(no_closed)                   # settled accounts = good sign

    # ── Loan structure ───────────────────────────────────────────────────────
    + 0.15 * zscore(loan_tenure)                 # longer term = more time-at-risk
    + 0.15 * zscore(max_credit_exp)              # high max exposure = more systemic risk

    # ── DPD anchors (post-event — drive realistic target distribution) ───────
    + 0.80 * (dpd30 > 0).astype(float)
    + 1.40 * (dpd60 > 0).astype(float)
    + 2.00 * (dpd90 > 0).astype(float)

    # ── Irreducible noise ────────────────────────────────────────────────────
    + np.random.normal(0, 0.55, N)
)

prob_default = expit(log_odds)
default      = (np.random.uniform(0, 1, N) < prob_default).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 10. APPLICATION DATES
# ─────────────────────────────────────────────────────────────────────────────
app_dates = random_dates('2023-01-01', '2025-12-31', N)

# ─────────────────────────────────────────────────────────────────────────────
# 11. ASSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    'Customer_ID'                         : range(N),
    'Age'                                 : ages,
    'Income_INR'                          : income,
    'Employment_Years'                    : employment_years,
    'Marital_Status'                      : marital,
    'Education_Level'                     : edu,
    'Credit_History_Length'               : credit_hist_len,
    'Outstanding_Loans'                   : outstanding,
    'Loan_Amount'                         : loan_amount,
    'Loan_Tenure_Months'                  : loan_tenure,
    'Savings_Account_Balance'             : savings,
    'Checking_Account_Balance'            : checking,
    'No_of_Open_Accounts'                 : no_open,
    'No_of_Closed_Accounts'               : no_closed,
    'Total_Credit_Limit'                  : total_credit_limit,
    'Total_Current_Balance'               : total_current_balance,
    'Credit_Utilization_Ratio'            : credit_util_ratio,
    'No_of_Inquiries_6M'                  : inq_6m,
    'No_of_Inquiries_12M'                 : inq_12m,
    'DPD_30'                              : dpd30,
    'DPD_60'                              : dpd60,
    'DPD_90'                              : dpd90,
    'Worst_Current_Status'                : worst_status,
    'Months_Since_Most_Recent_Delinquency': months_since_delinq,
    'Max_Credit_Exposure'                 : max_credit_exp,
    'Oldest_Trade_Open_Months'            : oldest_trade,
    'Newest_Trade_Open_Months'            : newest_trade,
    'Default'                             : default,
    'Application_Date'                    : app_dates,
})

# ─────────────────────────────────────────────────────────────────────────────
# 12. SAVE & VALIDATION REPORT
# ─────────────────────────────────────────────────────────────────────────────
out = 'data/s08/credit_data_v2.csv'
df.to_csv(out, index=False)

print(f"✅  Saved {len(df):,} rows → {out}")
print(f"\n{'═'*60}")
print(f"  Overall default rate : {df['Default'].mean():.3f}  ({df['Default'].sum()} defaults)")
print(f"{'═'*60}")

checks = {
    # variable: (group_col, group_labels_ordered_low_to_high_risk, expected_direction)
    'Education_Level (↑ edu → ↓ default)': (
        'Education_Level', ['Postgraduate', 'Graduate', 'High School'], 'ascending'
    ),
}

print("\n── Default rate by Education ──────────────────────────────────")
print(df.groupby('Education_Level')['Default'].mean().round(3).sort_values())

print("\n── Default rate by Income quartile (Q1=low → Q4=high) ─────────")
df['_iq'] = pd.qcut(df['Income_INR'], 4, labels=['Q1 Low','Q2','Q3','Q4 High'])
print(df.groupby('_iq', observed=True)['Default'].mean().round(3))

print("\n── Default rate by Employment quartile ─────────────────────────")
df['_eq'] = pd.qcut(df['Employment_Years'].astype(float)+np.random.uniform(0,0.01,N),
                    4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
print(df.groupby('_eq', observed=True)['Default'].mean().round(3))

print("\n── Default rate by Credit Utilisation tertile ──────────────────")
df['_ut'] = pd.qcut(df['Credit_Utilization_Ratio'], 3, labels=['Low','Mid','High'])
print(df.groupby('_ut', observed=True)['Default'].mean().round(3))

print("\n── Default rate by Marital Status ──────────────────────────────")
print(df.groupby('Marital_Status')['Default'].mean().round(3).sort_values())

print("\n── Default rate by Savings quartile ────────────────────────────")
df['_sq'] = pd.qcut(df['Savings_Account_Balance'], 4, labels=['Q1 Low','Q2','Q3','Q4 High'])
print(df.groupby('_sq', observed=True)['Default'].mean().round(3))

print("\n── Default rate by Debt-to-Income tertile ──────────────────────")
dti = df['Loan_Amount'] / (df['Income_INR'] + 1e-6)
df['_dti'] = pd.qcut(dti, 3, labels=['Low','Mid','High'])
print(df.groupby('_dti', observed=True)['Default'].mean().round(3))

print("\n── Default rate by Inquiries 12M ───────────────────────────────")
df['_inq'] = pd.cut(df['No_of_Inquiries_12M'], bins=[-1,0,2,5,99],
                    labels=['0','1-2','3-5','6+'])
print(df.groupby('_inq', observed=True)['Default'].mean().round(3))

print("\n── Default rate by Credit History quartile ─────────────────────")
df['_ch'] = pd.qcut(df['Credit_History_Length'].astype(float)+np.random.uniform(0,0.01,N),
                    4, labels=['Q1 Short','Q2','Q3','Q4 Long'], duplicates='drop')
print(df.groupby('_ch', observed=True)['Default'].mean().round(3))

print("\n── Default rate by Outstanding Loans ───────────────────────────")
print(df.groupby('Outstanding_Loans')['Default'].mean().round(3))

print("\n── Default rate by Worst DPD Status ────────────────────────────")
print(df.groupby('Worst_Current_Status')['Default'].mean().round(3).sort_values(ascending=False))

# drop temp cols
df.drop(columns=[c for c in df.columns if c.startswith('_')], inplace=True)
df.to_csv(out, index=False)
print(f"\n{'═'*60}")
print(f"  Final CSV (clean): {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"{'═'*60}")