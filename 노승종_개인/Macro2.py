import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
#  ★ 파라미터 설정 (단독 실행 시 여기서만 수정하세요)
# ══════════════════════════════════════════════════════════
N = 20    # 직전 N 영업일 변동성 계산 기간  ← 두 번째 코드 기준
P = 4     # Quantile 분류 개수 (Q1=저변동성, QP=고변동성)
# ══════════════════════════════════════════════════════════

ESI_FILE   = r'C:\Users\intern9\Timepolio_qt\DATA\ESI_지표.xlsx'
KOSPI_FILE = r'C:\Users\intern9\Timepolio_qt\DATA\코스피_시종가.xlsx'
OUT_FILE   = r'C:\Users\intern9\Timepolio_qt\DATA\분위수별_회귀분석결과_변동성.xlsx'


# ──────────────────────────────────────────────────────────
# [데이터 로드]
#   KOSPI : Date / Close(종가) / Open(시가)
#   ESI   : Date / ESI
# ──────────────────────────────────────────────────────────
def load_data(kospi_file, esi_file):
    kospi = pd.read_excel(kospi_file, header=13, usecols=[0, 1, 2])
    kospi.columns = ['Date', 'Close', 'Open']
    kospi = kospi.dropna(subset=['Date'])
    kospi['Date']  = pd.to_datetime(kospi['Date'])
    kospi['Close'] = pd.to_numeric(kospi['Close'], errors='coerce')
    kospi['Open']  = pd.to_numeric(kospi['Open'],  errors='coerce')
    kospi = kospi.dropna().sort_values('Date').reset_index(drop=True)

    esi = pd.read_excel(esi_file, header=6, usecols=[0, 1])
    esi.columns = ['Date', 'ESI']
    esi = esi.dropna(subset=['Date'])
    esi['Date'] = pd.to_datetime(esi['Date'])
    esi['ESI']  = pd.to_numeric(esi['ESI'], errors='coerce')
    esi = esi.dropna().sort_values('Date').reset_index(drop=True)

    return kospi, esi


# ──────────────────────────────────────────────────────────
# [수익률 계산]
#   종가 수익률 : (Close_t - Close_{t-1}) / Close_{t-1}
#   시가 수익률 : (Open_t  - Close_{t-1}) / Close_{t-1}  ← 전일종가 기준 갭
#   장중 수익률 : (Close_t - Open_t)      / Open_t       ← 시가매수 종가매도
# ──────────────────────────────────────────────────────────
def calc_returns(kospi):
    df = kospi.copy()
    df['Close_Return']    = df['Close'].pct_change()
    df['Open_Return']     = (df['Open']  - df['Close'].shift(1)) / df['Close'].shift(1)
    df['Intraday_Return'] = (df['Close'] - df['Open'])           / df['Open']
    df = df.iloc[1:].reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────
# [Delta_ESI 계산]
#   ESI_t1 = A일 직전 가장 최근 ESI (A-1 시점)
#   ESI_t2 = ESI_t1 바로 앞 ESI    (A-2 시점)
#   Delta_ESI     = ESI_t1 - ESI_t2
#   Delta_ESI_pos = max(ΔEsi, 0)  : ESI 상승 변화분
#   Delta_ESI_neg = min(ΔEsi, 0)  : ESI 하락 변화분
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
    merged['Delta_ESI']     = merged['ESI'] - merged['ESI_prev']
    merged['Delta_ESI_pos'] = merged['Delta_ESI'].clip(lower=0)
    merged['Delta_ESI_neg'] = merged['Delta_ESI'].clip(upper=0)
    merged = merged.dropna(subset=['Delta_ESI']).reset_index(drop=True)
    return merged


# ──────────────────────────────────────────────────────────
# [내부 함수] 분위수별 다중회귀
#   Y  : return_col ('Close_Return' 또는 'Open_Return')
#   X1 : Delta_ESI_pos  (ESI 양의 변화분)
#   X2 : Delta_ESI_neg  (ESI 음의 변화분)
# ──────────────────────────────────────────────────────────
def _run_multiple_regression_per_quantile(df_reg, P, group_col,
                                          range_col, return_col):
    rows = []
    for q_label in sorted(df_reg[group_col].unique()):
        sub = df_reg[df_reg[group_col] == q_label].copy()
        n = len(sub)

        q_min  = sub[range_col].min()
        q_max  = sub[range_col].max()
        q_mean = sub[range_col].mean()

        if n < 10:
            print(f"  [{q_label}] 관측치 부족 ({n}개) → 건너뜀")
            continue

        Y  = sub[return_col].values
        X1 = sub['Delta_ESI_pos'].values
        X2 = sub['Delta_ESI_neg'].values
        X  = sm.add_constant(np.column_stack([X1, X2]))

        try:
            model   = sm.OLS(Y, X).fit()
            const   = model.params[0]
            b_pos   = model.params[1]
            b_neg   = model.params[2]
            p_const = model.pvalues[0]
            p_pos   = model.pvalues[1]
            p_neg   = model.pvalues[2]
            r2      = model.rsquared
            adj_r2  = model.rsquared_adj
            f_stat  = model.fvalue
            f_pval  = model.f_pvalue
        except Exception as e:
            print(f"  [{q_label}] 회귀 오류: {e}")
            continue

        # ── 변동성 구간을 % 단위로 표시 (연환산 포함) ────────
        vol_lo_pct      = q_min  * 100
        vol_hi_pct      = q_max  * 100
        vol_lo_ann      = q_min  * np.sqrt(252) * 100
        vol_hi_ann      = q_max  * np.sqrt(252) * 100
        vol_mean_pct    = q_mean * 100
        vol_mean_ann    = q_mean * np.sqrt(252) * 100

        rows.append({
            '분위수'               : q_label,
            '관측치 수'            : n,
            f'{range_col}(일간%) 구간'   : f"[{vol_lo_pct:.3f}%, {vol_hi_pct:.3f}%]",
            f'{range_col}(연환산%) 구간' : f"[{vol_lo_ann:.1f}%, {vol_hi_ann:.1f}%]",
            f'{range_col} 평균(일간%)'   : round(vol_mean_pct, 4),
            f'{range_col} 평균(연환산%)' : round(vol_mean_ann, 2),
            '상수(Intercept)'      : round(const,  6),
            'p_상수'               : round(p_const, 4),
            'Beta_ESI_pos(양변화)' : round(b_pos,   6),
            'p_ESI_pos'            : round(p_pos,   4),
            'Beta_ESI_neg(음변화)' : round(b_neg,   6),
            'p_ESI_neg'            : round(p_neg,   4),
            'R²'                   : round(r2,      4),
            'Adj_R²'               : round(adj_r2,  4),
            'F통계량'              : round(f_stat,  4),
            'F_p값'                : round(f_pval,  4),
        })

        sig_pos = '***' if p_pos < 0.01 else ('**' if p_pos < 0.05 else ('*' if p_pos < 0.1 else ''))
        sig_neg = '***' if p_neg < 0.01 else ('**' if p_neg < 0.05 else ('*' if p_neg < 0.1 else ''))
        print(f"\n  [{q_label}]  n={n}")
        print(f"    변동성 구간(일간):  [{vol_lo_pct:.3f}%, {vol_hi_pct:.3f}%]  "
              f"평균={vol_mean_pct:.3f}%")
        print(f"    변동성 구간(연환산): [{vol_lo_ann:.1f}%, {vol_hi_ann:.1f}%]  "
              f"평균={vol_mean_ann:.1f}%")
        print(f"    Beta_pos={b_pos:.6f} (p={p_pos:.4f}){sig_pos}")
        print(f"    Beta_neg={b_neg:.6f} (p={p_neg:.4f}){sig_neg}")
        print(f"    R²={r2:.4f}  Adj_R²={adj_r2:.4f}  F={f_stat:.4f}(p={f_pval:.4f})")

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────
# [추가 함수] 분위수별 시가매수→종가매도 전략 성과
#   승률        = Intraday_Return > 0 인 비율
#   평균 수익률 = Intraday_Return 평균
#   이항검정    = 승률이 50%를 유의하게 초과하는지 (단측)
# ──────────────────────────────────────────────────────────
def print_intraday_strategy_stats(df_reg, group_col, range_col,
                                  out_file=None, sheet_name='장중전략_성과'):
    from scipy.stats import binomtest

    print("\n" + "═"*72)
    print(f"{'분위수별  시가매수 → 종가매도  전략 성과':^72}")
    print(f"{'전략 수익률 = (당일 종가 - 당일 시가) / 당일 시가':^72}")
    print(f"{'Q1 = 저변동성 국면  →  Q' + str(df_reg[group_col].nunique()) + ' = 고변동성 국면':^72}")
    print("═"*72)
    print(f"  {'분위수':<6}  {'관측치':>6}  {'변동성평균(연환산%)':>18}  "
          f"{'승률':>9}  {'평균수익률':>10}  {'승':>5}  {'패':>5}")
    print("  " + "-"*75)

    rows = []
    for q_label in sorted(df_reg[group_col].unique()):
        sub  = df_reg[df_reg[group_col] == q_label].copy()
        n    = len(sub)

        intra    = sub['Intraday_Return']
        wins     = int((intra > 0).sum())
        losses   = int((intra <= 0).sum())
        win_rate = wins / n if n > 0 else np.nan
        avg_ret  = intra.mean()
        # 변동성 연환산 평균
        vol_ann  = sub[range_col].mean() * np.sqrt(252) * 100

        binom_p = binomtest(wins, n, p=0.5, alternative='greater').pvalue
        star = ('***' if binom_p < 0.01 else
                '**'  if binom_p < 0.05 else
                '*'   if binom_p < 0.1  else ' ')

        print(f"  {str(q_label):<6}  {n:>6}  {vol_ann:>17.1f}%  "
              f"{win_rate:>7.1%}{star}  {avg_ret:>+10.4%}  {wins:>5}  {losses:>5}")

        rows.append({
            '분위수'              : q_label,
            '관측치 수'           : n,
            f'{range_col} 연환산% 평균': round(vol_ann, 2),
            '승 횟수'             : wins,
            '패 횟수'             : losses,
            '승률'                : round(win_rate, 4),
            '평균 수익률'         : round(avg_ret,  6),
            '이항검정 p값'        : round(binom_p,  4),
        })

    print("  " + "-"*75)
    print("  ※ 승률 옆 별표: 이항검정(H0: 승률=50%, 단측)  * p<0.1  ** p<0.05  *** p<0.01")

    df_stats = pd.DataFrame(rows)

    if out_file:
        with pd.ExcelWriter(out_file, engine='openpyxl',
                            mode='a' if _file_exists(out_file) else 'w') as writer:
            df_stats.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"\n✅ 저장 완료: {out_file}  (시트: {sheet_name})")

    return df_stats


# ──────────────────────────────────────────────────────────
# [핵심 함수] 변동성 기반 Quantile 분류 + 다중회귀분석
#   → import 시 이 함수만 호출하면 됨
#
#   df_input   : Delta_ESI / Delta_ESI_pos / Delta_ESI_neg /
#                Close_Return / Open_Return / Intraday_Return 컬럼이 있는 DataFrame
#   N          : 직전 N 영업일 변동성 계산 기간
#   P          : Quantile 개수
#   return_col : 'Close_Return'(종가) 또는 'Open_Return'(시가)
#   out_file   : 결과 저장 경로 (None이면 저장 안 함)
#   sheet_name : 엑셀 저장 시 시트명
#   title      : 출력 헤더 타이틀
# ──────────────────────────────────────────────────────────
def run_vol_quantile_regression(df_input, N, P,
                                return_col='Close_Return',
                                out_file=None, sheet_name='Sheet1',
                                title=''):
    """
    반환값: (회귀분석 요약 DataFrame, 분위수 라벨이 붙은 df_reg)
    """
    df = df_input.copy()

    # ── [핵심 변경] 분위수 기준: 수익률 평균 → 수익률 변동성 ──
    #   Vol_N = 직전 N 영업일 종가수익률의 표준편차 (당일 제외)
    df['Vol_N'] = df['Close_Return'].shift(1).rolling(window=N).std(ddof=1)
    # ─────────────────────────────────────────────────────────

    df = df.dropna(subset=['Vol_N', 'Close_Return', 'Open_Return',
                            'Intraday_Return', 'Delta_ESI_pos', 'Delta_ESI_neg']).copy()

    # 변동성 기반 Quantile 분류 (Q1=저변동성, QP=고변동성)
    df['Quantile'] = pd.qcut(
        df['Vol_N'],
        q=P,
        labels=[f'Q{i+1}' for i in range(P)]
    )

    print(f"\n변동성 분위수 분포 (P={P},  Q1=저변동성 → Q{P}=고변동성):")
    print(df['Quantile'].value_counts().sort_index())

    df_reg = df.dropna(subset=['Quantile']).copy()
    print(f"\n회귀 분석 가능한 총 관측치: {len(df_reg):,}개")

    return_label = '종가(Close)' if return_col == 'Close_Return' else '시가(Open) ← 전일종가 기준 갭'
    header_title = (title if title else
                    f'직전 N={N} 영업일 변동성 기준 분위수  (P={P})'
                    f'  |  Q1=저변동성  Q{P}=고변동성')

    print("\n" + "═"*72)
    print(f"{'분위수별 다중 OLS 회귀분석 결과':^72}")
    print(f"{'Y: ' + return_label + '  |  X1: ESI 양변화  X2: ESI 음변화':^72}")
    print(f"{header_title:^72}")
    print("═"*72)

    df_result = _run_multiple_regression_per_quantile(
        df_reg, P,
        group_col='Quantile', range_col='Vol_N',
        return_col=return_col
    )

    print("\n" + "═"*72)
    print("요약 테이블")
    print("═"*72)
    print(df_result.to_string(index=False))

    if out_file:
        with pd.ExcelWriter(out_file, engine='openpyxl',
                            mode='a' if _file_exists(out_file) else 'w') as writer:
            df_result.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"\n✅ 결과 저장 완료: {out_file}  (시트: {sheet_name})")

    return df_result, df_reg


def _file_exists(path):
    import os
    return os.path.exists(path)


# ──────────────────────────────────────────────────────────
# 단독 실행 시 (python Macro_vol.py)
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os

    kospi, esi = load_data(KOSPI_FILE, ESI_FILE)

    print(f"ESI   데이터: {esi['Date'].min().date()} ~ {esi['Date'].max().date()}  ({len(esi):,}일)")
    print(f"KOSPI 데이터: {kospi['Date'].min().date()} ~ {kospi['Date'].max().date()}  ({len(kospi):,}영업일)")

    kospi_ret    = calc_returns(kospi)
    kospi_merged = calc_delta_esi(kospi_ret, esi)

    print(f"\nN={N} 기준 분석 가능 기간: {kospi_merged['Date'].iloc[0].date()} ~ {kospi_merged['Date'].iloc[-1].date()}")
    print(f"  총 {len(kospi_merged):,}개 영업일")

    if os.path.exists(OUT_FILE):
        os.remove(OUT_FILE)

    # ── [1/3] 종가 수익률 회귀 ─────────────────────────────
    print("\n\n" + "★"*72)
    print("  [1/3]  Y = 종가 수익률  (Close Return)")
    print("★"*72)
    df_close, df_reg_close = run_vol_quantile_regression(
        df_input=kospi_merged, N=N, P=P,
        return_col='Close_Return',
        out_file=OUT_FILE, sheet_name='종가_수익률',
    )

    # ── [2/3] 시가 수익률 회귀 ─────────────────────────────
    print("\n\n" + "★"*72)
    print("  [2/3]  Y = 시가 수익률  (Open Return: 전일 종가 → 당일 시가 갭)")
    print("★"*72)
    df_open, df_reg_open = run_vol_quantile_regression(
        df_input=kospi_merged, N=N, P=P,
        return_col='Open_Return',
        out_file=OUT_FILE, sheet_name='시가_수익률',
    )

    # ── [3/3] 시가매수→종가매도 전략 성과 ──────────────────
    # Quantile 기준이 동일하므로 df_reg_close 재사용
    print("\n\n" + "★"*72)
    print("  [3/3]  시가매수 → 종가매도  전략  승률 & 평균수익률")
    print("★"*72)
    df_intraday = print_intraday_strategy_stats(
        df_reg=df_reg_close,
        group_col='Quantile',
        range_col='Vol_N',
        out_file=OUT_FILE,
        sheet_name='장중전략_성과',
    )

    print("\n\n" + "="*72)
    print("  모든 분석 완료")
    print(f"  저장 경로: {OUT_FILE}")
    print("="*72)
    #python macro/Macro2.py