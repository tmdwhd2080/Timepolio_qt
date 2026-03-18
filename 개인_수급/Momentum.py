import pandas as pd
import numpy as np

# ============================================================
# ★ 하드코딩 파라미터 (여기서만 변경하세요)
# ============================================================
FILE_PATH      = r"C:\Users\intern9\Timepolio_qt\DATA\탐폴_데이터.xlsx"
BENCHMARK_PATH = r"C:\Users\intern9\Timepolio_qt\DATA\코스피_코스닥_종가.xlsx"
N_DAYS         = 2          # 순매수 합산 기간 (n일)
# ============================================================

# ── 1. 탐폴 데이터 로드
raw = pd.read_excel(FILE_PATH, header=None)

N_STOCKS = 300
COLS_PER = 7
codes     = raw.iloc[7, 1::COLS_PER].values[:N_STOCKS]
data_rows = raw.iloc[14:]
dates     = pd.to_datetime(data_rows.iloc[:, 0]).values

# ── 2. Long 형식 DataFrame 구성
records = []
for i in range(N_STOCKS):
    base = 1 + i * COLS_PER
    tmp = pd.DataFrame({
        "date"  : dates,
        "code"  : codes[i],
        "close" : pd.to_numeric(data_rows.iloc[:, base + 0].values, errors="coerce"),
        "mktcap": pd.to_numeric(data_rows.iloc[:, base + 1].values, errors="coerce"),
        "fininv": pd.to_numeric(data_rows.iloc[:, base + 3].values, errors="coerce"),
        "indiv" : pd.to_numeric(data_rows.iloc[:, base + 6].values, errors="coerce"),
    })
    records.append(tmp)

df = pd.concat(records, ignore_index=True)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "code"]).reset_index(drop=True)

# ── 3. 전일 종가 대비 수익률 (다음 영업일)
df["return_next"] = df.groupby("code")["close"].transform(
    lambda x: x.shift(-1) / x - 1
)

# ── 4. 시가총액 순위 → 대형주 / 중형주 분류
df["mktcap_rank"] = df.groupby("date")["mktcap"].rank(ascending=False, method="first")
df["cap_type"] = np.where(
    df["mktcap_rank"] <= 100, "대형주",
    np.where(df["mktcap_rank"] <= 300, "중형주", None)
)
df = df[df["cap_type"].notna()].copy()

# ── 5. 개인 + 금융투자 순매수 합산
df["net_buy"] = df["indiv"].fillna(0) + df["fininv"].fillna(0)

daily_agg = (
    df.groupby(["date", "cap_type"])
    .agg(
        net_buy_sum      = ("net_buy",     "sum"),
        return_next_mean = ("return_next", "mean"),
    )
    .reset_index()
    .sort_values(["cap_type", "date"])
)

# ── 6. 코스피 벤치마크 로드 및 다음날 수익률 계산
bm_raw = pd.read_excel(BENCHMARK_PATH, header=None)
bm = bm_raw.iloc[14:, [0, 2]].copy()          # col0=날짜, col2=코스피 종가지수
bm.columns = ["date", "kospi"]
bm["date"]  = pd.to_datetime(bm["date"])
bm["kospi"] = pd.to_numeric(bm["kospi"], errors="coerce")
bm = bm.sort_values("date").reset_index(drop=True)
bm["bm_return_next"] = bm["kospi"].shift(-1) / bm["kospi"] - 1  # 다음날 벤치마크 수익률
bm = bm[["date", "bm_return_next"]]

# ── 7. 그룹 수익률에 벤치마크 수익률 합산 (초과수익률 계산)
daily_agg = daily_agg.merge(bm, on="date", how="left")
daily_agg["excess_return"] = daily_agg["return_next_mean"] - daily_agg["bm_return_next"]

# ── 8. n일 롤링 합산 시그널
results = []
for cap, grp in daily_agg.groupby("cap_type"):
    grp = grp.copy().reset_index(drop=True)
    grp["signal"]     = grp["net_buy_sum"].rolling(N_DAYS, min_periods=N_DAYS).sum()
    grp["signal_pos"] = grp["signal"] > 0
    results.append(grp)

result_df = pd.concat(results, ignore_index=True)
result_df = result_df.dropna(subset=["signal", "excess_return"])

# ── 9. 결과 출력
sep = "=" * 70
print(f"\n{sep}")
print(f"  분석 결과  |  N = {N_DAYS}일  순매수 합산 기간  (벤치마크: 코스피)")
print(sep)
print(f"  시그널    : (개인 순매수대금 + 금융투자 순매수대금)의 {N_DAYS}일 합계 부호")
print(f"  초과수익률: 종목 동일가중 평균수익률 − 코스피 다음날 수익률")
print(f"  대형주    : 당일 시가총액 순위 1~100위")
print(f"  중형주    : 당일 시가총액 순위 101~300위")
print(f"{sep}\n")

for cap in ["대형주", "중형주"]:
    sub = result_df[result_df["cap_type"] == cap]
    pos = sub[sub["signal_pos"] == True]
    neg = sub[sub["signal_pos"] == False]

    def stats(grp, label):
        n = len(grp)
        if n == 0:
            return f"  {label}: 데이터 없음"
        prob  = (grp["excess_return"] > 0).mean()      # 벤치마크 초과 확률
        avg_r = grp["excess_return"].mean()             # 평균 초과수익률
        return (
            f"  {label:<14} | 관측 {n:>4}일 | "
            f"BM 초과 확률: {prob*100:>6.2f}% | "
            f"평균 초과수익률: {avg_r*100:>+7.4f}%"
        )

    print(f"■ {cap}")
    print(stats(pos, "시그널 양(+)"))
    print(stats(neg, "시그널 음(-)"))
    print()

# ── 10. 초과수익률 분포
print(sep)
print(f"  [분포]  Q25 / 중앙값 / Q75  (단위: %, 초과수익률 기준)")
print(sep)
for cap in ["대형주", "중형주"]:
    sub = result_df[result_df["cap_type"] == cap]
    for sign, label in [(True, "양(+)"), (False, "음(-)")]:
        grp = sub[sub["signal_pos"] == sign]["excess_return"]
        if len(grp) == 0:
            continue
        q25, med, q75 = grp.quantile([0.25, 0.5, 0.75]) * 100
        print(
            f"  {cap} / 시그널 {label}  |  "
            f"Q25={q25:+.4f}%  중앙={med:+.4f}%  Q75={q75:+.4f}%"
        )
print()
#python 개인_수급/Momentum.py