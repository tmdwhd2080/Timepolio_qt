import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
#  ★ 파라미터 설정 (단독 실행 시 여기서만 수정하세요)
# ══════════════════════════════════════════════════════════
N = 3     # 직전 N 영업일 평균수익률 계산 기간
P = 4     # Quantile 분류 개수
# ══════════════════════════════════════════════════════════

ESI_FILE   = r'C:\Users\intern9\Timepolio_qt\DATA\ESI_지표.xlsx'
KOSPI_FILE = r'C:\Users\intern9\Timepolio_qt\DATA\코스피_장기.xlsx'
OUT_FILE   = r'C:\Users\intern9\Timepolio_qt\DATA\분위수별_회귀분석결과.xlsx'


# ──────────────────────────────────────────────────────────
# [핵심 함수] 수익률 기반 Quantile 분류 + 회귀분석
#   → import 시 이 함수만 호출하면 됨
#
#   df_input : 이미 필터링된 A일 후보 DataFrame
#              (필수 컬럼: 'index'(원본 kospi 순번), 'Return', 'Delta_ESI')
#   kospi    : 원본 전체 KOSPI DataFrame (Avg_Ret_N 계산용)
#   N        : 직전 N 영업일 평균수익률 기간
#   P        : Quantile 개수
#   out_file : 결과 저장 경로 (None이면 저장 안 함)
#   title    : 출력 헤더 타이틀 (호출처에서 맥락 설명용)
# ──────────────────────────────────────────────────────────
def run_return_quantile_regression(df_input, kospi, N, P,
                                   out_file=None, title=''):
    """
    반환값: 회귀분석 요약 DataFrame
    """
    from Macro2 import run_regression_per_quantile

    # Avg_Ret_N을 전체 kospi 기준으로 계산 (rolling은 전체 시계열에서)
    kospi = kospi.copy()
    kospi['Avg_Ret_N'] = kospi['Return'].shift(1).rolling(window=N).mean()

    # df_input의 'index' 컬럼(원본 kospi 순번)으로 Avg_Ret_N 매핑
    avg_map = kospi['Avg_Ret_N'].to_dict()   # {원본순번: Avg_Ret_N}
    df = df_input.copy()
    df['Avg_Ret_N'] = df['index'].map(avg_map)

    # Avg_Ret_N이 NaN인 행 제거 (N보다 앞선 행)
    df = df.dropna(subset=['Avg_Ret_N', 'Return', 'Delta_ESI']).copy()

    # 수익률 기반 Quantile 분류
    df['Quantile'] = pd.qcut(
        df['Avg_Ret_N'],
        q=P,
        labels=[f'Q{i+1}' for i in range(P)]
    )

    print(f"\n수익률 분위수 분포 (P={P}):")
    print(df['Quantile'].value_counts().sort_index())

    df_reg = df.dropna(subset=['Delta_ESI', 'Return', 'Quantile']).copy()
    print(f"\n회귀 분석 가능한 총 관측치: {len(df_reg):,}개")

    header_title = title if title else f'직전 N={N} 영업일 평균수익률 기준 분위수  (P={P})'
    print("\n" + "═"*70)
    print(f"{'분위수별 OLS 회귀분석 결과':^70}")
    print(f"{'Y: A일 수익률  |  X: ESI 변화량 (A-2 → A-1 영업일)':^70}")
    print(f"{header_title:^70}")
    print("═"*70)

    df_result = run_regression_per_quantile(
        df_reg, P,
        group_col='Quantile', range_col='Avg_Ret_N',
        annualize=False
    )

    print("\n" + "═"*70)
    print("요약 테이블")
    print("═"*70)
    print(df_result.to_string(index=False))

    if out_file:
        df_result.to_excel(out_file, index=False)
        print(f"\n✅ 결과 저장 완료: {out_file}")

    return df_result


# ──────────────────────────────────────────────────────────
# 단독 실행 시 (python Macro.py)
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    from Macro2 import load_data, calc_delta_esi

    esi, kospi = load_data(ESI_FILE, KOSPI_FILE)

    print(f"ESI   데이터: {esi['Date'].min().date()} ~ {esi['Date'].max().date()}  ({len(esi):,}일)")
    print(f"KOSPI 데이터: {kospi['Date'].min().date()} ~ {kospi['Date'].max().date()}  ({len(kospi):,}영업일)")

    # 전체 A일 생성 (Avg_Ret_N 기준)
    kospi_work = kospi.copy()
    kospi_work['Avg_Ret_N'] = kospi_work['Return'].shift(1).rolling(window=N).mean()
    kospi_a = kospi_work.dropna(subset=['Return', 'Avg_Ret_N']).reset_index(drop=False)

    print(f"\nN={N} 영업일 기준  A 범위: {len(kospi_a):,}개 영업일")
    print(f"  첫 A일: {kospi_a['Date'].iloc[0].date()}  (데이터 시작 후 {N+1}번째 영업일)")
    print(f"  끝 A일: {kospi_a['Date'].iloc[-1].date()}")

    # ESI 변화량 계산
    kospi_a = calc_delta_esi(kospi_a, kospi, esi)

    run_return_quantile_regression(
        df_input=kospi_a,
        kospi=kospi,
        N=N,
        P=P,
        out_file=OUT_FILE
    )