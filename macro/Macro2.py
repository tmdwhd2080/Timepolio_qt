import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
#  ★ 파라미터 설정 (단독 실행 시 여기서만 수정하세요)
# ══════════════════════════════════════════════════════════
N = 20    # 직전 N 영업일 변동성 계산 기간
P = 4     # Quantile 분류 개수 (QP = 고변동성 국면)
# ══════════════════════════════════════════════════════════

ESI_FILE   = r'C:\Users\intern9\Timepolio_qt\DATA\ESI_지표.xlsx'
KOSPI_FILE = r'C:\Users\intern9\Timepolio_qt\DATA\코스피_장기.xlsx'
OUT_FILE   = r'C:\Users\intern9\Timepolio_qt\DATA\분위수별_회귀분석결과_변동성.xlsx'


# ──────────────────────────────────────────────────────────
# [공통 함수] 데이터 로드
# ──────────────────────────────────────────────────────────
def load_data(esi_file, kospi_file):
    esi = pd.read_excel(esi_file, header=None, skiprows=7)
    esi.columns = ['Date', 'ESI']
    esi = (esi
           .dropna(subset=['Date', 'ESI'])
           .assign(Date=lambda d: pd.to_datetime(d['Date']),
                   ESI =lambda d: pd.to_numeric(d['ESI'], errors='coerce'))
           .dropna()
           .sort_values('Date')
           .reset_index(drop=True))

    kospi = pd.read_excel(kospi_file, header=None, skiprows=14)
    kospi.columns = ['Date', 'Close', 'Volume', '_']
    kospi = (kospi[['Date', 'Close']]
             .dropna(subset=['Date', 'Close'])
             .assign(Date =lambda d: pd.to_datetime(d['Date']),
                     Close=lambda d: pd.to_numeric(d['Close'], errors='coerce'))
             .dropna()
             .sort_values('Date')
             .reset_index(drop=True))

    kospi['Return'] = kospi['Close'].pct_change()
    return esi, kospi


# ──────────────────────────────────────────────────────────
# [공통 함수] ESI 변화량 계산 (A-2 → A-1 영업일)
# ──────────────────────────────────────────────────────────
def calc_delta_esi(kospi_a, kospi, esi):
    esi_sorted = esi.sort_values('Date').reset_index(drop=True)

    def get_esi_on_or_before(target_date):
        idx = esi_sorted['Date'].searchsorted(target_date, side='right') - 1
        return esi_sorted.loc[idx, 'ESI'] if idx >= 0 else np.nan

    delta_esi = []
    for _, row in kospi_a.iterrows():
        pos = int(row['index'])
        if pos < 2:
            delta_esi.append(np.nan)
            continue
        d_a1 = kospi.loc[pos - 1, 'Date']
        d_a2 = kospi.loc[pos - 2, 'Date']
        esi_a1 = get_esi_on_or_before(d_a1)
        esi_a2 = get_esi_on_or_before(d_a2)
        if np.isnan(esi_a1) or np.isnan(esi_a2):
            delta_esi.append(np.nan)
        else:
            delta_esi.append(esi_a1 - esi_a2)

    kospi_a = kospi_a.copy()
    kospi_a['Delta_ESI'] = delta_esi
    return kospi_a


# ──────────────────────────────────────────────────────────
# [공통 함수] OLS 회귀분석 수행 및 결과 반환
# ──────────────────────────────────────────────────────────
def run_regression_per_quantile(df_reg, P, group_col, range_col, range_scale=100, annualize=False):
    """
    df_reg      : 회귀 대상 DataFrame (Delta_ESI, Return, Quantile 컬럼 포함)
    P           : Quantile 개수
    group_col   : Quantile 레이블 컬럼명
    range_col   : 구간 출력용 컬럼명 (Vol_N 또는 Avg_Ret_N)
    range_scale : 구간 값에 곱할 스케일 (기본 100 → %)
    annualize   : True이면 연환산 변동성도 출력
    """
    results = []

    for q_label in [f'Q{i+1}' for i in range(P)]:
        sub = df_reg[df_reg[group_col] == q_label].copy()
        n_obs = len(sub)

        if n_obs < 5:
            print(f"\n[{q_label}] 관측치 부족 ({n_obs}개) → 스킵")
            continue

        x = sub['Delta_ESI'].values
        y = sub['Return'].values

        slope, intercept, r_val, p_val, se = stats.linregress(x, y)
        r2      = r_val ** 2
        t_stat  = slope / se if se != 0 else np.nan
        avg_ret = np.mean(y)
        std_ret = np.std(y, ddof=1)

        y_hat  = intercept + slope * x
        df_res = n_obs - 2
        rse    = np.sqrt(np.sum((y - y_hat) ** 2) / df_res) if df_res > 0 else np.nan

        lo = sub[range_col].min() * range_scale
        hi = sub[range_col].max() * range_scale

        row = dict(
            Quantile          = q_label,
            N_관측치           = n_obs,
            구간               = f"[{lo:.3f}%, {hi:.3f}%]",
            절편               = round(intercept, 6),
            기울기              = round(slope, 6),
            t_통계량            = round(t_stat, 4),
            p_value           = round(p_val, 4),
            유의여부_5pct      = "★ 유의" if p_val < 0.05 else "-",
            R_squared         = round(r2, 4),
            상관계수_R         = round(r_val, 4),
            잔차표준오차        = round(rse, 6),
            평균수익률_pct     = round(avg_ret * 100, 4),
            수익률표준편차_pct = round(std_ret * 100, 4),
        )
        if annualize:
            row['연환산변동성_구간'] = f"[{lo*np.sqrt(252):.1f}%, {hi*np.sqrt(252):.1f}%]"

        results.append(row)

        print(f"\n{'─'*60}")
        print(f" {q_label}  |  관측치: {n_obs}개  |  구간: [{lo:.3f}%, {hi:.3f}%]", end="")
        if annualize:
            print(f"  (연환산: [{lo*np.sqrt(252):.1f}%, {hi*np.sqrt(252):.1f}%])")
        else:
            print()
        print(f"{'─'*60}")
        print(f"  절편(Intercept)   : {intercept:>12.6f}")
        print(f"  기울기(Slope)     : {slope:>12.6f}")
        print(f"  t 통계량          : {t_stat:>12.4f}")
        print(f"  p-value           : {p_val:>12.4f}  {'★ 유의 (5%)' if p_val < 0.05 else ''}")
        print(f"  R²                : {r2:>12.4f}")
        print(f"  상관계수 R        : {r_val:>12.4f}")
        print(f"  잔차표준오차(RSE) : {rse:>12.6f}")
        print(f"  평균수익률        : {avg_ret*100:>11.4f}%")
        print(f"  수익률 표준편차   : {std_ret*100:>11.4f}%")

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────
# [핵심 함수] 변동성 기반 Quantile 분류 테이블 생성
#   → import 시 이 함수만 호출하면 됨
# ──────────────────────────────────────────────────────────
def build_vol_quantile_table(esi, kospi, N, P):
    """
    반환값:
      kospi_a : Vol_N, Quantile, Delta_ESI 컬럼이 추가된 DataFrame
                (원본 kospi 영업일 순번 'index' 컬럼 포함)
    """
    kospi = kospi.copy()
    kospi['Vol_N'] = kospi['Return'].shift(1).rolling(window=N).std(ddof=1)

    kospi_a = kospi.dropna(subset=['Return', 'Vol_N']).reset_index(drop=False)

    kospi_a = kospi_a.copy()
    kospi_a['Quantile'] = pd.qcut(
        kospi_a['Vol_N'],
        q=P,
        labels=[f'Q{i+1}' for i in range(P)]
    )

    kospi_a = calc_delta_esi(kospi_a, kospi, esi)

    return kospi_a


# ──────────────────────────────────────────────────────────
# 단독 실행 시 (python Macro2.py)
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    esi, kospi = load_data(ESI_FILE, KOSPI_FILE)

    print(f"ESI   데이터: {esi['Date'].min().date()} ~ {esi['Date'].max().date()}  ({len(esi):,}일)")
    print(f"KOSPI 데이터: {kospi['Date'].min().date()} ~ {kospi['Date'].max().date()}  ({len(kospi):,}영업일)")

    kospi_a = build_vol_quantile_table(esi, kospi, N, P)

    print(f"\nN={N} 영업일 기준  A 범위: {len(kospi_a):,}개 영업일")
    print(f"  첫 A일: {kospi_a['Date'].iloc[0].date()}  (데이터 시작 후 {N+1}번째 영업일)")
    print(f"  끝 A일: {kospi_a['Date'].iloc[-1].date()}")
    print(f"\n변동성 분위수 분포 (P={P}):")
    print(kospi_a['Quantile'].value_counts().sort_index())

    df_reg = kospi_a.dropna(subset=['Delta_ESI', 'Return', 'Quantile']).copy()
    print(f"\n회귀 분석 가능한 총 관측치: {len(df_reg):,}개")

    print("\n" + "═"*70)
    print(f"{'분위수별 OLS 회귀분석 결과':^70}")
    print(f"{'Y: A일 수익률  |  X: ESI 변화량 (A-2 → A-1 영업일)':^70}")
    print(f"{'직전 N=' + str(N) + ' 영업일 변동성 기준 분위수  (P=' + str(P) + ')':^70}")
    print(f"{'Q1 = 저변동성 국면  →  Q' + str(P) + ' = 고변동성 국면':^70}")
    print("═"*70)

    df_result = run_regression_per_quantile(
        df_reg, P,
        group_col='Quantile', range_col='Vol_N',
        annualize=True
    )

    print("\n" + "═"*70)
    print("요약 테이블")
    print("═"*70)
    print(df_result.to_string(index=False))

    df_result.to_excel(OUT_FILE, index=False)
    print(f"\n✅ 결과 저장 완료: {OUT_FILE}")