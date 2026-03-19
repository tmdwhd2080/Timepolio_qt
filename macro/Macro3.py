import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from Macro2 import load_data, build_vol_quantile_table, run_regression_per_quantile
from Macro  import run_return_quantile_regression

# ══════════════════════════════════════════════════════════
#  ★ 파라미터 설정 (여기서만 수정하세요)
# ══════════════════════════════════════════════════════════
N_VOL = 20   # [1단계] 변동성 계산 기간 (영업일)
P_VOL = 4    # [1단계] 변동성 Quantile 개수 → 최고 분위(Q{P_VOL})만 추출

N_RET = 3    # [2단계] 평균수익률 계산 기간 (영업일)
P_RET = 3    # [2단계] 수익률 Quantile 개수
# ══════════════════════════════════════════════════════════

ESI_FILE   = r'C:\Users\intern9\Timepolio_qt\DATA\ESI_지표.xlsx'
KOSPI_FILE = r'C:\Users\intern9\Timepolio_qt\DATA\코스피_장기.xlsx'
OUT_FILE   = r'C:\Users\intern9\Timepolio_qt\DATA\분위수별_회귀분석결과_combined.xlsx'

# ──────────────────────────────────────────────────────────
# 1. 데이터 로드
# ──────────────────────────────────────────────────────────
esi, kospi = load_data(ESI_FILE, KOSPI_FILE)

print(f"ESI   데이터: {esi['Date'].min().date()} ~ {esi['Date'].max().date()}  ({len(esi):,}일)")
print(f"KOSPI 데이터: {kospi['Date'].min().date()} ~ {kospi['Date'].max().date()}  ({len(kospi):,}영업일)")

# ──────────────────────────────────────────────────────────
# 2. [1단계] 변동성 기반 Quantile 분류 (Macro2 활용)
#    → 전체 A일을 N_VOL 영업일 변동성으로 P_VOL개 분위 분류
#    → ESI 변화량(Delta_ESI)도 이 단계에서 함께 계산됨
# ──────────────────────────────────────────────────────────
print("\n" + "═"*70)
print(f"{'[1단계] 변동성 기반 Quantile 분류':^70}")
print(f"{'N_VOL=' + str(N_VOL) + ' 영업일  |  P_VOL=' + str(P_VOL) + '개 분위':^70}")
print(f"{'→ 최고 변동성 구간 Q' + str(P_VOL) + ' 표본만 추출':^70}")
print("═"*70)

kospi_a_vol = build_vol_quantile_table(esi, kospi, N_VOL, P_VOL)

print(f"\n전체 A일 수: {len(kospi_a_vol):,}개")
print(f"  첫 A일: {kospi_a_vol['Date'].iloc[0].date()}")
print(f"  끝 A일: {kospi_a_vol['Date'].iloc[-1].date()}")

print(f"\n변동성 분위수 분포 (P_VOL={P_VOL}):")
print(kospi_a_vol['Quantile'].value_counts().sort_index())

print(f"\n변동성 구간 요약:")
for q_label in [f'Q{i+1}' for i in range(P_VOL)]:
    sub = kospi_a_vol[kospi_a_vol['Quantile'] == q_label]
    lo = sub['Vol_N'].min() * 100
    hi = sub['Vol_N'].max() * 100
    marker = " ← 이 구간 추출" if q_label == f'Q{P_VOL}' else ""
    print(f"  {q_label}: [{lo:.3f}%, {hi:.3f}%]"
          f"  (연환산: [{lo*np.sqrt(252):.1f}%, {hi*np.sqrt(252):.1f}%]){marker}")

# ──────────────────────────────────────────────────────────
# 3. 최고 변동성 Quantile(Q{P_VOL}) 표본만 추출
# ──────────────────────────────────────────────────────────
high_vol_label  = f'Q{P_VOL}'
df_high_vol     = kospi_a_vol[kospi_a_vol['Quantile'] == high_vol_label].copy()
df_high_vol_reg = df_high_vol.dropna(subset=['Delta_ESI', 'Return']).copy()

vol_lo = df_high_vol_reg['Vol_N'].min() * 100
vol_hi = df_high_vol_reg['Vol_N'].max() * 100

print(f"\n추출된 고변동성({high_vol_label}) 표본: {len(df_high_vol_reg):,}개")
print(f"  일별변동성 구간  : [{vol_lo:.3f}%, {vol_hi:.3f}%]")
print(f"  연환산변동성 구간: [{vol_lo*np.sqrt(252):.1f}%, {vol_hi*np.sqrt(252):.1f}%]")

# ──────────────────────────────────────────────────────────
# 4. [2단계] 고변동성 표본 내에서 수익률 기반 Quantile 분류
#            + OLS 회귀분석 (Macro 활용)
# ──────────────────────────────────────────────────────────
print("\n" + "═"*70)
print(f"{'[2단계] 고변동성 표본 내 수익률 기반 Quantile 분류 + 회귀분석':^70}")
print(f"{'N_RET=' + str(N_RET) + ' 영업일  |  P_RET=' + str(P_RET) + '개 분위':^70}")
print("═"*70)

title_str = (f"고변동성({high_vol_label}) 표본 내 "
             f"직전 N_RET={N_RET} 영업일 평균수익률 기준 분위수  (P_RET={P_RET})")

df_result = run_return_quantile_regression(
    df_input = df_high_vol_reg,
    kospi    = kospi,
    N        = N_RET,
    P        = P_RET,
    out_file = OUT_FILE,
    title    = title_str
)

print(f"\n✅ 전체 분석 완료")
#python macro/Macro3.py