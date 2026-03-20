import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import binomtest
import warnings
import os
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
#  ★ 파라미터 설정 (여기서만 수정하세요)
# ══════════════════════════════════════════════════════════
N_RET = 3    # 시가 평균수익률 기준 분위수 계산 기간 (영업일)
N_VOL = 10   # 변동성 기준 분위수 계산 기간 (영업일)
Q     = 4    # Quantile 개수
# ══════════════════════════════════════════════════════════

ESI_FILE   = r'C:\Users\intern9\Timepolio_qt\DATA\ESI_지표.xlsx'
KOSPI_FILE = r'C:\Users\intern9\Timepolio_qt\DATA\코스피_시종가.xlsx'
OUT_FILE   = r'C:\Users\intern9\Timepolio_qt\DATA\분석결과.xlsx'


# ──────────────────────────────────────────────────────────
# 1. 데이터 로드
#    KOSPI 컬럼 순서: DATE | 시가(Open) | 종가(Close)
# ──────────────────────────────────────────────────────────
def load_data(kospi_file, esi_file):
    kospi = pd.read_excel(kospi_file, header=13, usecols=[0, 1, 2])
    kospi.columns = ['Date', 'Open', 'Close']
    kospi['Date']  = pd.to_datetime(kospi['Date'])
    kospi['Open']  = pd.to_numeric(kospi['Open'],  errors='coerce')
    kospi['Close'] = pd.to_numeric(kospi['Close'], errors='coerce')
    kospi = (kospi.dropna()
                  .sort_values('Date')
                  .reset_index(drop=True))

    esi = pd.read_excel(esi_file, header=6, usecols=[0, 1])
    esi.columns = ['Date', 'ESI']
    esi['Date'] = pd.to_datetime(esi['Date'])
    esi['ESI']  = pd.to_numeric(esi['ESI'], errors='coerce')
    esi = (esi.dropna()
              .sort_values('Date')
              .reset_index(drop=True))

    return kospi, esi


# ──────────────────────────────────────────────────────────
# 2. 수익률 계산
#   Open_Return     : (Open_T  - Close_{T-1}) / Close_{T-1}
#   Close_Return    : (Close_T - Close_{T-1}) / Close_{T-1}
#   Intraday_Return : (Close_T - Open_T)      / Open_T
# ──────────────────────────────────────────────────────────
def calc_returns(kospi):
    df = kospi.copy()
    df['Close_Return']    = df['Close'].pct_change()
    df['Open_Return']     = (df['Open']  - df['Close'].shift(1)) / df['Close'].shift(1)
    df['Intraday_Return'] = (df['Close'] - df['Open'])           / df['Open']
    df = df.iloc[1:].reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────
# 3. Delta_ESI 계산
#   Delta_ESI = ESI_{T-1} - ESI_{T-2}  (부호 반영)
# ──────────────────────────────────────────────────────────
def calc_delta_esi(kospi_ret, esi):
    esi_s = esi.sort_values('Date').copy()
    esi_s['ESI_prev'] = esi_s['ESI'].shift(1)

    merged = pd.merge_asof(
        kospi_ret.sort_values('Date'),
        esi_s[['Date', 'ESI', 'ESI_prev']].rename(columns={'Date': 'ESI_Date'}),
        left_on='Date', right_on='ESI_Date',
        direction='backward'
    )
    merged['Delta_ESI'] = merged['ESI'] - merged['ESI_prev']
    merged = merged.dropna(subset=['Delta_ESI']).reset_index(drop=True)
    return merged


# ──────────────────────────────────────────────────────────
# 4. 단일회귀: Y = a + b * Delta_ESI  (OLS)
# ──────────────────────────────────────────────────────────
def _ols(y, x):
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit()
    return {
        'beta'   : res.params[1],
        't_stat' : res.tvalues[1],
        'p_value': res.pvalues[1],
        'r2'     : res.rsquared,
        'adj_r2' : res.rsquared_adj,
    }


# ──────────────────────────────────────────────────────────
# 5. 장중전략 성과 계산 (내부 헬퍼)
#    sub     : 분위수 내 표본 DataFrame
#    label   : 출력용 레이블 ('전체' / 'ESI↑(양)' / 'ESI↓(음)')
# ──────────────────────────────────────────────────────────
def _intraday_stats(sub, label):
    """
    반환: dict  (장중전략 성과 지표)
    """
    n        = len(sub)
    intra    = sub['Intraday_Return']
    wins     = int((intra > 0).sum())
    losses   = int((intra <= 0).sum())
    win_rate = wins / n if n > 0 else np.nan
    avg_ret  = intra.mean() if n > 0 else np.nan

    if n >= 1:
        binom_p = binomtest(wins, n, p=0.5, alternative='greater').pvalue
        star    = ('***' if binom_p < 0.01 else
                   '**'  if binom_p < 0.05 else
                   '*'   if binom_p < 0.1  else ' ')
    else:
        binom_p = np.nan
        star    = ' '

    print(f"      [{label:8s}]  n={n:4d}  "
          f"승률={win_rate:.1%}{star}  "
          f"평균수익률={avg_ret:+.4%}  "
          f"승={wins}  패={losses}  "
          f"(이항검정 p={binom_p:.4f})" if not np.isnan(binom_p)
          else f"      [{label:8s}]  n={n}  데이터 부족")

    return {
        'ESI구분'     : label,
        'n'           : n,
        '승 횟수'     : wins,
        '패 횟수'     : losses,
        '승률'        : round(win_rate, 4) if not np.isnan(win_rate) else np.nan,
        '평균 수익률' : round(avg_ret,  6) if not np.isnan(avg_ret)  else np.nan,
        '이항검정 p'  : round(binom_p,  4) if not np.isnan(binom_p) else np.nan,
    }


# ──────────────────────────────────────────────────────────
# 6. 분위수별 회귀 + 장중전략 성과 계산
#    ★ 장중전략: 전체 / ESI 양(+) / ESI 음(-) 3구간으로 분리
# ──────────────────────────────────────────────────────────
def _analyze_quantiles(df_reg, group_col, regime_col, regime_type, Q):
    reg_rows   = []
    intra_rows = []

    for q_label in sorted(df_reg[group_col].unique()):
        sub = df_reg[df_reg[group_col] == q_label].copy()
        n   = len(sub)

        r_min  = sub[regime_col].min()
        r_max  = sub[regime_col].max()
        r_mean = sub[regime_col].mean()

        if n < 5:
            print(f"  [{q_label}] 관측치 부족 ({n}개) → 건너뜀")
            continue

        # ── 회귀분석 ──────────────────────────────────────
        y = sub['Open_Return'].values
        x = sub['Delta_ESI'].values
        try:
            res = _ols(y, x)
        except Exception as e:
            print(f"  [{q_label}] 회귀 오류: {e}")
            continue

        # 출력용 범위 문자열
        if regime_type == 'vol':
            range_str     = f"[{r_min*100:.3f}%, {r_max*100:.3f}%]"
            range_ann_str = f"[{r_min*np.sqrt(252)*100:.1f}%, {r_max*np.sqrt(252)*100:.1f}%]"
            mean_str      = f"{r_mean*100:.3f}% (연환산 {r_mean*np.sqrt(252)*100:.1f}%)"
        else:
            range_str     = f"[{r_min*100:.3f}%, {r_max*100:.3f}%]"
            range_ann_str = ""
            mean_str      = f"{r_mean*100:.4f}%"

        sig = ('***' if res['p_value'] < 0.01 else
               '**'  if res['p_value'] < 0.05 else
               '*'   if res['p_value'] < 0.1  else '')

        print(f"\n  [{q_label}]  n={n}  범위={range_str}", end="")
        if range_ann_str:
            print(f"  연환산={range_ann_str}", end="")
        print(f"\n    평균={mean_str}")
        print(f"    Beta={res['beta']:.6f}  t={res['t_stat']:.4f}  "
              f"p={res['p_value']:.4f}{sig}  R²={res['r2']:.4f}")

        reg_row = {
            '분위수'   : q_label,
            '관측치 수': n,
            '범위'     : range_str,
            '평균'     : round(r_mean, 6),
            'Beta'     : round(res['beta'],    6),
            't_통계량' : round(res['t_stat'],  4),
            'p_value'  : round(res['p_value'], 4),
            '유의여부' : ('***' if res['p_value'] < 0.01 else
                          '**'  if res['p_value'] < 0.05 else
                          '*'   if res['p_value'] < 0.1  else '-'),
            'R²'       : round(res['r2'],     4),
            'Adj_R²'   : round(res['adj_r2'], 4),
        }
        if regime_type == 'vol':
            reg_row['범위(연환산)'] = range_ann_str
        reg_rows.append(reg_row)

        # ── 장중전략: 전체 / ESI 양 / ESI 음 ─────────────
        sub_pos = sub[sub['Delta_ESI'] >  0]   # ESI 상승일
        sub_neg = sub[sub['Delta_ESI'] <= 0]   # ESI 하락(+보합)일

        print(f"    ┌─ 장중전략 (시가매수→종가매도) 분위수 내 ESI 방향별 분리")
        stats_all = _intraday_stats(sub,     '전체    ')
        stats_pos = _intraday_stats(sub_pos, 'ESI↑(양)')
        stats_neg = _intraday_stats(sub_neg, 'ESI↓(음)')
        print(f"    └─ ※ 이항검정: H0=승률50%, 단측  *p<0.1  **p<0.05  ***p<0.01")

        for s in [stats_all, stats_pos, stats_neg]:
            intra_rows.append({
                '분위수'      : q_label,
                'ESI 구분'    : s['ESI구분'].strip(),
                '관측치 수'   : s['n'],
                '범위'        : range_str,
                '승 횟수'     : s['승 횟수'],
                '패 횟수'     : s['패 횟수'],
                '승률'        : s['승률'],
                '평균 수익률' : s['평균 수익률'],
                '이항검정 p'  : s['이항검정 p'],
            })

    return pd.DataFrame(reg_rows), pd.DataFrame(intra_rows)


# ──────────────────────────────────────────────────────────
# 7. [핵심 함수 A] 시가 평균수익률 기준 Quantile 분석
# ──────────────────────────────────────────────────────────
def run_ret_quantile(df_merged, N_RET, Q, out_file=None):
    df = df_merged.copy()

    df['AvgOpen_N'] = df['Open_Return'].shift(1).rolling(window=N_RET).mean()
    df = df.dropna(subset=['AvgOpen_N', 'Open_Return',
                            'Intraday_Return', 'Delta_ESI']).copy()

    df['Quantile'] = pd.qcut(
        df['AvgOpen_N'],
        q=Q,
        labels=[f'Q{i+1}' for i in range(Q)]
    )
    df_reg = df.dropna(subset=['Quantile']).copy()

    print("\n" + "★"*72)
    print(f"  [분석 A]  시가 평균수익률 기준 Quantile  (N={N_RET}일, Q={Q})")
    print(f"  Y: T일 시가 수익률  |  X: ESI 변화량(T-2→T-1)")
    print(f"  Q1 = 직전 수익률 낮음  →  Q{Q} = 직전 수익률 높음")
    print("★"*72)

    print(f"\n분위수 분포:")
    print(df_reg['Quantile'].value_counts().sort_index().to_string())
    print(f"\n총 분석 관측치: {len(df_reg):,}개")

    df_reg_res, df_intra_res = _analyze_quantiles(
        df_reg, 'Quantile', 'AvgOpen_N', 'ret', Q
    )

    print("\n" + "─"*72)
    print("  회귀 요약")
    print("─"*72)
    print(df_reg_res.to_string(index=False))
    print("\n" + "─"*72)
    print("  장중전략 요약  (시가매수→종가매도, ESI 방향별)")
    print("─"*72)
    print(df_intra_res.to_string(index=False))
    print("  ※ 승률 별표: 이항검정(H0: 승률=50%, 단측)  *p<0.1  **p<0.05  ***p<0.01")

    if out_file:
        mode = 'a' if os.path.exists(out_file) else 'w'
        with pd.ExcelWriter(out_file, engine='openpyxl', mode=mode) as w:
            df_reg_res.to_excel(w,   sheet_name=f'A_회귀_수익률N{N_RET}',   index=False)
            df_intra_res.to_excel(w, sheet_name=f'A_장중전략_수익률N{N_RET}', index=False)
        print(f"\n✅ 저장: {out_file}")

    return df_reg_res, df_intra_res


# ──────────────────────────────────────────────────────────
# 8. [핵심 함수 B] 변동성 기준 Quantile 분석
# ──────────────────────────────────────────────────────────
def run_vol_quantile(df_merged, N_VOL, Q, out_file=None):
    df = df_merged.copy()

    df['Vol_N'] = df['Close_Return'].shift(1).rolling(window=N_VOL).std(ddof=1)
    df = df.dropna(subset=['Vol_N', 'Open_Return',
                             'Intraday_Return', 'Delta_ESI']).copy()

    df['Quantile'] = pd.qcut(
        df['Vol_N'],
        q=Q,
        labels=[f'Q{i+1}' for i in range(Q)]
    )
    df_reg = df.dropna(subset=['Quantile']).copy()

    print("\n" + "★"*72)
    print(f"  [분석 B]  변동성 기준 Quantile  (N={N_VOL}일, Q={Q})")
    print(f"  Y: T일 시가 수익률  |  X: ESI 변화량(T-2→T-1)")
    print(f"  Q1 = 저변동성 국면  →  Q{Q} = 고변동성 국면")
    print("★"*72)

    print(f"\n분위수 분포:")
    print(df_reg['Quantile'].value_counts().sort_index().to_string())
    print(f"\n총 분석 관측치: {len(df_reg):,}개")

    df_reg_res, df_intra_res = _analyze_quantiles(
        df_reg, 'Quantile', 'Vol_N', 'vol', Q
    )

    print("\n" + "─"*72)
    print("  회귀 요약")
    print("─"*72)
    print(df_reg_res.to_string(index=False))
    print("\n" + "─"*72)
    print("  장중전략 요약  (시가매수→종가매도, ESI 방향별)")
    print("─"*72)
    print(df_intra_res.to_string(index=False))
    print("  ※ 승률 별표: 이항검정(H0: 승률=50%, 단측)  *p<0.1  **p<0.05  ***p<0.01")

    if out_file:
        mode = 'a' if os.path.exists(out_file) else 'w'
        with pd.ExcelWriter(out_file, engine='openpyxl', mode=mode) as w:
            df_reg_res.to_excel(w,   sheet_name=f'B_회귀_변동성N{N_VOL}',   index=False)
            df_intra_res.to_excel(w, sheet_name=f'B_장중전략_변동성N{N_VOL}', index=False)
        print(f"\n✅ 저장: {out_file}")

    return df_reg_res, df_intra_res


# ──────────────────────────────────────────────────────────
# 단독 실행
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':

    kospi, esi = load_data(KOSPI_FILE, ESI_FILE)

    print(f"KOSPI 데이터: {kospi['Date'].min().date()} ~ "
          f"{kospi['Date'].max().date()}  ({len(kospi):,}영업일)")
    print(f"ESI   데이터: {esi['Date'].min().date()}  ~ "
          f"{esi['Date'].max().date()}  ({len(esi):,}일)")

    kospi_ret    = calc_returns(kospi)
    kospi_merged = calc_delta_esi(kospi_ret, esi)

    print(f"\n분석 가능 기간: {kospi_merged['Date'].iloc[0].date()} ~ "
          f"{kospi_merged['Date'].iloc[-1].date()}  "
          f"({len(kospi_merged):,}영업일)")

    if os.path.exists(OUT_FILE):
        os.remove(OUT_FILE)

    run_ret_quantile(kospi_merged, N_RET, Q, out_file=OUT_FILE)
    run_vol_quantile(kospi_merged, N_VOL, Q, out_file=OUT_FILE)

    print("\n\n" + "="*72)
    print("  모든 분석 완료")
    print(f"  저장 경로: {OUT_FILE}")
    print("="*72)
# python macro/Macro.py