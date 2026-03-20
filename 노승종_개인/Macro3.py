import requests
import pandas as pd
import time

# ══════════════════════════════════════════════════════════
#  ★ 인증 정보 (만료 시 브라우저에서 다시 복사)
# ══════════════════════════════════════════════════════════
COOKIES = {
    "t": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJqdGkiOiJmbEZWV2VES0lwWXhBdm5vTXU4M1dTV0VlYWJ0VG9ScyIsImV4cCI6MTc3Mzk5ODc5MX0.P0nC4-Y5pHh_0kb9sAC6BvpG-d7Ed5TRp4btCKcCIs8",
}

HEADERS = {
    "accept"          : "application/json;version=2.0",
    "accept-language" : "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "origin"          : "https://platform.worldquantbrain.com",
    "referer"         : "https://platform.worldquantbrain.com/",
    "user-agent"      : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
}
# ══════════════════════════════════════════════════════════

BASE_URL = "https://api.worldquantbrain.com/data-fields"

# ── 파라미터 (필요 시 수정) ────────────────────────────────
PARAMS = {
    "delay"          : 1,
    "instrumentType" : "EQUITY",
    "region"         : "USA",
    "universe"       : "TOP3000",
    "limit"          : 20,    # 한 번에 가져올 개수 (최대값 확인 후 늘려도 됨)
}


def crawl_all_fields():
    all_records = []
    offset      = 0

    # 첫 요청으로 전체 개수 파악
    resp = requests.get(
        BASE_URL,
        headers=HEADERS,
        cookies=COOKIES,
        params={**PARAMS, "offset": 0},
    )

    if resp.status_code == 401:
        print("❌ 인증 만료 — 브라우저에서 쿠키를 다시 복사하세요.")
        return pd.DataFrame()
    if resp.status_code != 200:
        print(f"❌ 요청 실패: {resp.status_code}  {resp.text[:300]}")
        return pd.DataFrame()

    data  = resp.json()
    total = data.get("count", data.get("total", None))
    print(f"✅ 첫 요청 성공 — 전체 field 수: {total}")

    # 응답 구조 확인 (어떤 키에 데이터가 있는지)
    print(f"   응답 최상위 키: {list(data.keys())}")

    # 레코드 리스트 키 자동 탐색
    record_key = None
    for key in ["results", "data", "fields", "items"]:
        if key in data and isinstance(data[key], list):
            record_key = key
            break
    if record_key is None:
        print("❌ 레코드 리스트 키를 찾지 못했습니다. 응답 구조를 확인하세요:")
        print(data)
        return pd.DataFrame()

    print(f"   레코드 키: '{record_key}'\n")

    # 전체 페이지 순회
    while True:
        resp = requests.get(
            BASE_URL,
            headers=HEADERS,
            cookies=COOKIES,
            params={**PARAMS, "offset": offset},
        )

        if resp.status_code == 401:
            print("❌ 인증 만료 — 브라우저에서 쿠키를 다시 복사하세요.")
            break
        if resp.status_code != 200:
            print(f"❌ 오류: {resp.status_code}")
            break

        records = resp.json().get(record_key, [])

        if not records:
            print(f"   offset={offset} 에서 데이터 없음 → 수집 완료")
            break

        all_records.extend(records)
        print(f"   수집 중 ... offset={offset:>5}  "
              f"이번 배치={len(records):>3}개  "
              f"누적={len(all_records):>5}개", end="")

        if total:
            print(f"  / {total}  ({len(all_records)/total*100:.1f}%)")
        else:
            print()

        # 종료 조건
        if total and len(all_records) >= total:
            break
        if len(records) < PARAMS["limit"]:
            break

        offset += PARAMS["limit"]
        time.sleep(0.3)    # 서버 부하 방지

    return pd.DataFrame(all_records)


if __name__ == "__main__":
    df = crawl_all_fields()

    if df.empty:
        print("수집된 데이터가 없습니다.")
    else:
        print(f"\n✅ 최종 수집 완료: {len(df)}개 field")
        print(df.head(3).to_string())

        # 저장
        out_path = "worldquant_fields.xlsx"
        df.to_excel(out_path, index=False)
        print(f"\n💾 저장 완료: {out_path}")
#python macro/Macro3.py