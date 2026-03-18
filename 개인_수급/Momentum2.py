import pandas as pd
import numpy as np

# ============================================================
# ★ 하드코딩 파라미터 (여기서만 변경하세요)
# ============================================================
FILE_PATH      = r"C:\Users\intern9\Timepolio_qt\DATA\탐폴_데이터.xlsx"
BENCHMARK_PATH = r"C:\Users\intern9\Timepolio_qt\DATA\코스피_코스닥_종가.xlsx"
N_DAYS = 2   # 순매수 시그널 합산 기간 (n일)
K_DAYS = 1   # 이후 성과 측정 기간 (k일)
# ============================================================

# ── 1. 탐폴 데이터 로드
raw = pd.read_excel(FILE_PATH, header=None)

N_STOCKS = 300
COLS_PER = 7
# offset: 0=종가, 1=시가총액, 2=기관, 3=금융투자, 4=외국인, 5=거래량, 6=개인
codes     = raw.iloc[7, 1::COLS_PER].values[:N_STOCKS]
data_rows = raw.iloc[14:]
dates     = pd.to_datetime(data_rows.iloc[:, 0]).values

# ── 2. Long 형식 DataFrame 구성
records = []
for i in range(N_STOCKS):
    base = 1 + i * COLS_PER
    tmp = pd.DataFrame({
        "date"   : dates,
        "code"   : codes[i],
        "close"  : pd.to_numeric(data_rows.iloc[:, base + 0].values, errors="coerce"),
        "mktcap" : pd.to_numeric(data_rows.iloc[:, base + 1].values, errors="coerce"),
        "instit" : pd.to_numeric(data_rows.iloc[:, base + 2].values, errors="coerce"),
        "foreign": pd.to_numeric(data_rows.iloc[:, base + 4].values, errors="coerce"),
        "indiv"  : pd.to_numeric(data_rows.iloc[:, base + 6].values, errors="coerce"),
    })
    records.append(tmp)

df = pd.concat(records, ignore_index=True)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "code"]).reset_index(drop=True)

# ── 3. 다음날 수익률
df["return_1d"] = df.groupby("code")["close"].transform(
    lambda x: x.shift(-1) / x - 1
)

# ── 4. 시가총액 순위 → 대형주 / 중형주
df["mktcap_rank"] = df.groupby("date")["mktcap"].rank(ascending=False, method="first")
df["cap_type"] = np.where(
    df["mktcap_rank"] <= 100, "대형주",
    np.where(df["mktcap_rank"] <= 300, "중형주", None)
)
df = df[df["cap_type"].notna()].copy()

# ── 5. 외인 / 전체(외인+기관+개인) 순매수
df["net_foreign"] = df["foreign"].fillna(0)
df["net_total"]   = df["foreign"].fillna(0) + df["instit"].fillna(0) + df["indiv"].fillna(0)

daily_agg = (
    df.groupby(["date", "cap_type"])
    .agg(
        foreign_sum = ("net_foreign", "sum"),
        total_sum   = ("net_total",   "sum"),
        ret_mean    = ("return_1d",   "mean"),
    )
    .reset_index()
    .sort_values(["cap_type", "date"])
)

# ── 6. 코스피 벤치마크
bm_raw = pd.read_excel(BENCHMARK_PATH, header=None)
bm = bm_raw.iloc[14:, [0, 2]].copy()
bm.columns = ["date", "kospi"]
bm["date"]  = pd.to_datetime(bm["date"])
bm["kospi"] = pd.to_numeric(bm["kospi"], errors="coerce")
bm = bm.sort_values("date").reset_index(drop=True)
bm["bm_ret"] = bm["kospi"].shift(-1) / bm["kospi"] - 1
bm = bm[["date", "bm_ret"]]

daily_agg = daily_agg.merge(bm, on="date", how="left")
daily_agg["excess_1d"] = daily_agg["ret_mean"] - daily_agg["bm_ret"]

# ── 7. n일 시그널 + k일 누적 초과수익률
results = []
for cap, grp in daily_agg.groupby("cap_type"):
    grp = grp.copy().reset_index(drop=True)
    grp["sig_foreign"] = grp["foreign_sum"].rolling(N_DAYS, min_periods=N_DAYS).sum()
    grp["sig_total"]   = grp["total_sum"].rolling(N_DAYS, min_periods=N_DAYS).sum()
    # t일 기준 이후 k일간 누적 초과수익률
    grp["excess_kd"] = grp["excess_1d"].rolling(K_DAYS).sum().shift(-(K_DAYS - 1))
    results.append(grp)

result_df = pd.concat(results, ignore_index=True)
result_df = result_df.dropna(subset=["sig_foreign", "sig_total", "excess_kd"])

# ── 8. 결과 출력
CASES = [
    ("외인 양 / 전체 양",  True,  True),
    ("외인 양 / 전체 음",  True,  False),
    ("외인 음 / 전체 양",  False, True),
    ("외인 음 / 전체 음",  False, False),
]

sep = "=" * 76
print(f"\n{sep}")
print(f"  분석 결과  |  시그널 합산={N_DAYS}일  |  성과 측정={K_DAYS}일 누적  |  벤치마크: 코스피")
print(sep)
print(f"  외인 시그널 : 외국인 순매수대금 {N_DAYS}일 합계 부호")
print(f"  전체 시그널 : (외인 + 기관 + 개인) 순매수대금 {N_DAYS}일 합계 부호")
print(f"  초과수익률  : 종목 동일가중 평균 {K_DAYS}일 누적수익률 − 코스피 {K_DAYS}일 누적수익률")
print(sep)

for cap in ["대형주", "중형주"]:
    sub = result_df[result_df["cap_type"] == cap]
    print(f"\n  ■ {cap}")
    print(f"  {'─'*72}")
    print(f"  {'시그널 조합':<22} | {'관측':>5} | {'BM 초과 확률':>11} | {'평균 초과수익률':>14}")
    print(f"  {'─'*22}─{'─'*7}─{'─'*13}─{'─'*15}")

    for label, for_pos, tot_pos in CASES:
        for_mask = (sub["sig_foreign"] > 0) if for_pos else (sub["sig_foreign"] <= 0)
        tot_mask = (sub["sig_total"]   > 0) if tot_pos else (sub["sig_total"]   <= 0)
        grp = sub[for_mask & tot_mask]["excess_kd"]
        n   = len(grp)
        if n == 0:
            print(f"  {label:<22} | {'N/A':>5} | {'N/A':>11} | {'N/A':>14}")
            continue
        prob = (grp > 0).mean()
        avg  = grp.mean()
        print(
            f"  {label:<22} | {n:>5}일 | "
            f"{prob*100:>10.2f}% | {avg*100:>+13.4f}%"
        )

print(f"\n{sep}")
print(f"  [분포]  Q25 / 중앙값 / Q75  (단위: %, {K_DAYS}일 누적 초과수익률)")
print(sep)
for cap in ["대형주", "중형주"]:
    sub = result_df[result_df["cap_type"] == cap]
    print(f"\n  ■ {cap}")
    for label, for_pos, tot_pos in CASES:
        for_mask = (sub["sig_foreign"] > 0) if for_pos else (sub["sig_foreign"] <= 0)
        tot_mask = (sub["sig_total"]   > 0) if tot_pos else (sub["sig_total"]   <= 0)
        grp = sub[for_mask & tot_mask]["excess_kd"]
        if len(grp) == 0:
            continue
        q25, med, q75 = grp.quantile([0.25, 0.5, 0.75]) * 100
        print(
            f"  {label:<22} | "
            f"Q25={q25:+.4f}%  중앙={med:+.4f}%  Q75={q75:+.4f}%"
        )
print()
#python 개인_수급/Momentum2.py