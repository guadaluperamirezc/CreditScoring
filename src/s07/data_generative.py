import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
N = 2000

# ── helpers ──────────────────────────────────────────────────────────────────
def random_dates(start, end, n):
    start_ts = pd.Timestamp(start).value // 10**9
    end_ts   = pd.Timestamp(end).value   // 10**9
    return pd.to_datetime(
        np.random.randint(start_ts, end_ts, n), unit='s'
    ).strftime('%Y-%m-%d')

# ── base demographics ─────────────────────────────────────────────────────────
ages = np.random.randint(21, 70, N)
edu  = np.random.choice(
    ['High School', 'Graduate', 'Postgraduate'],
    N, p=[0.25, 0.50, 0.25]
)

# Income correlated with age + education
base_income = (ages - 20) * 3000
edu_boost   = np.where(edu == 'High School', 0,
              np.where(edu == 'Graduate',    150000, 300000))
income = (base_income + edu_boost +
          np.random.normal(0, 60000, N)).clip(100000, 5000000).astype(int)

employment_years = np.clip(
    (ages - 22) * np.random.uniform(0.5, 1.0, N) +
    np.random.normal(0, 2, N),
    0, 40
).astype(int)

marital = np.random.choice(
    ['Single', 'Married', 'Divorced', 'Widowed'],
    N, p=[0.35, 0.45, 0.15, 0.05]
)

# ── credit history ────────────────────────────────────────────────────────────
credit_hist_len = np.clip(
    employment_years * 0.8 + np.random.randint(1, 10, N),
    1, 30
).astype(int)

no_open   = np.random.randint(1, 8, N)
no_closed = np.random.randint(0, 10, N)

# ── loan details ──────────────────────────────────────────────────────────────
# Loan amount ~ income × factor
loan_factor  = np.random.uniform(0.5, 5.0, N)
loan_amount  = (income * loan_factor).clip(50000, 10000000).astype(int)
loan_tenure  = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240], N)

outstanding = np.random.randint(0, 5, N)

# ── account balances ──────────────────────────────────────────────────────────
savings  = (income * np.random.uniform(0.05, 2.0, N)).clip(0, 5000000).astype(int)
checking = (income * np.random.uniform(0.02, 0.5, N)).clip(0, 1000000).astype(int)

# ── credit utilisation (coherent) ────────────────────────────────────────────
total_credit_limit   = (income * np.random.uniform(0.5, 3.0, N)).clip(50000, 10000000).astype(int)
credit_util_ratio    = np.random.beta(2, 5, N).round(2)          # skewed low
total_current_balance = (total_credit_limit * credit_util_ratio).astype(int)

# ── inquiries ─────────────────────────────────────────────────────────────────
inq_6m  = np.random.poisson(1.2, N)
inq_12m = inq_6m + np.random.poisson(0.8, N)   # 12m always >= 6m

# ── delinquency / DPD ────────────────────────────────────────────────────────
# ~20 % of customers have some delinquency
dpd_mask = np.random.random(N) < 0.20

dpd30 = np.where(dpd_mask, np.random.randint(1, 5, N), 0)
dpd60 = np.where(dpd_mask, np.random.randint(0, dpd30.clip(1) + 1), 0)
dpd90 = np.where(dpd_mask, np.random.randint(0, dpd60.clip(1) + 1), 0)

worst_status = np.where(
    dpd90 > 0, 'DPD90+',
    np.where(dpd60 > 0, 'DPD60',
    np.where(dpd30 > 0, 'DPD30', 'Current'))
)

months_since_delinq = np.where(
    dpd_mask,
    np.random.randint(1, 36, N),
    np.nan
)

# ── max credit exposure & trade ages ─────────────────────────────────────────
max_credit_exp   = (total_credit_limit * np.random.uniform(1.0, 1.5, N)).astype(int)
oldest_trade     = credit_hist_len * 12 + np.random.randint(0, 12, N)
newest_trade     = np.random.randint(1, oldest_trade + 1)

# ── default label (target variable) ──────────────────────────────────────────
# Logistic-style: higher util, DPD, inquiries → higher default probability
default_score = (
    0.3 * credit_util_ratio +
    0.2 * (dpd30 > 0).astype(float) +
    0.3 * (dpd60 > 0).astype(float) +
    0.4 * (dpd90 > 0).astype(float) +
    0.05 * inq_12m / 10 +
    np.random.normal(0, 0.1, N)
)
default = (default_score > 0.30).astype(int)

# ── application dates ─────────────────────────────────────────────────────────
app_dates = random_dates('2023-01-01', '2025-12-31', N)

# ── assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    'Customer_ID':                  range(N),
    'Age':                          ages,
    'Income_INR':                   income,
    'Employment_Years':             employment_years,
    'Marital_Status':               marital,
    'Education_Level':              edu,
    'Credit_History_Length':        credit_hist_len,
    'Outstanding_Loans':            outstanding,
    'Loan_Amount':                  loan_amount,
    'Loan_Tenure_Months':           loan_tenure,
    'Savings_Account_Balance':      savings,
    'Checking_Account_Balance':     checking,
    'No_of_Open_Accounts':          no_open,
    'No_of_Closed_Accounts':        no_closed,
    'Total_Credit_Limit':           total_credit_limit,
    'Total_Current_Balance':        total_current_balance,
    'Credit_Utilization_Ratio':     credit_util_ratio,
    'No_of_Inquiries_6M':           inq_6m,
    'No_of_Inquiries_12M':          inq_12m,
    'DPD_30':                       dpd30,
    'DPD_60':                       dpd60,
    'DPD_90':                       dpd90,
    'Worst_Current_Status':         worst_status,
    'Months_Since_Most_Recent_Delinquency': months_since_delinq,
    'Max_Credit_Exposure':          max_credit_exp,
    'Oldest_Trade_Open_Months':     oldest_trade,
    'Newest_Trade_Open_Months':     newest_trade,
    'Default':                      default,
    'Application_Date':             app_dates,
})

out = 'credit_data.csv'
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows → {out}")
print(df.head(3).to_string())
print("\nDefault rate:", df['Default'].mean().round(3))
print("Worst status distribution:\n", df['Worst_Current_Status'].value_counts())