"""
WorldQuant Brain 데이터 필드 추천 봇
- Gemini 1.5 Flash API 사용
- 전략 설명 → 관련 데이터 필드 + 대용치 + 상세 설명 반환
"""

import os
import json
import ast
import pandas as pd
import google.generativeai as genai


GEMINI_API_KEY = "AIzaSyDWklWQY5xkY8hCz8D1iri-pJshnFzhAy4"   # https://aistudio.google.com/app/apikey
FIELDS_FILE    = r"C:\Users\intern9\Timepolio_qt\노승종_개인\worldquant_fields.xlsx"        



def load_fields(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # dict 형태 컬럼 → 이름만 추출
    def extract_name(val):
        if pd.isna(val):
            return ""
        if isinstance(val, dict):
            return val.get("name", "")
        try:
            d = ast.literal_eval(str(val))
            return d.get("name", "")
        except Exception:
            return str(val)

    df["category_name"]    = df["category"].apply(extract_name)
    df["subcategory_name"] = df["subcategory"].apply(extract_name)
    df["dataset_name"]     = df["dataset"].apply(extract_name)

    return df


def build_field_context(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        line = (
            f"[FIELD] id={row['id']} | "
            f"category={row['category_name']} | "
            f"subcategory={row['subcategory_name']} | "
            f"dataset={row['dataset_name']} | "
            f"type={row.get('type','')} | "
            f"description={row['description']}"
        )
        lines.append(line)
    return "\n".join(lines)



def init_gemini(api_key: str):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction="""
당신은 WorldQuant Brain 플랫폼의 퀀트 전략 전문가입니다.
사용자가 투자 전략을 설명하면, 아래 역할을 수행하세요.

[역할]
1. 전략 구현에 직접 사용 가능한 데이터 필드(primary fields)를 추천하세요.
2. 직접 데이터가 없을 경우 대용치(proxy fields)가 될 수 있는 필드도 추천하세요.
3. 각 필드에 대해 description을 바탕으로 해당 필드에 대해서 자세히 설명해주고, 왜 이 전략에 적합한지 한국어로 설명하세요.

[출력 형식] 반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요.
{
  "strategy_summary": "전략 요약 (1~2문장)",
  "primary_fields": [
    {
      "id": "필드 id",
      "category": "카테고리",
      "description": "원본 description",
      "reason": "이 필드에 대한 자세한 설명과 전략에 적합한 이유 (한국어, 3~4문장)"
    }
  ],
  "proxy_fields": [
    {
      "id": "필드 id",
      "category": "카테고리",
      "description": "원본 description",
      "reason": "이 필드에 대한 자세한 설명과 대용치로 활용 가능한 이유 (한국어, 3~4문장)"
    }
  ],
  "notes": "추가 조언 또는 주의사항 (한국어)"
}
""",
    )
    return model



def recommend_fields(model, field_context: str, strategy: str) -> dict:
    prompt = f"""
아래는 WorldQuant Brain에서 사용 가능한 전체 데이터 필드 목록입니다.

=== 데이터 필드 목록 ===
{field_context}

=== 사용자 전략 설명 ===
{strategy}

위 전략을 구현하기 위해 가장 적합한 primary 필드와 proxy 필드를 JSON으로 추천해주세요.
primary는 최대 8개, proxy는 최대 5개로 제한합니다.
"""
    response = model.generate_content(prompt)
    raw = response.text.strip()

    # 마크다운 코드블록 제거
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)



def print_result(result: dict):
    print("\n" + "═" * 70)
    print(f"  전략 요약: {result.get('strategy_summary', '')}")
    print("═" * 70)

    print("\n📌 [Primary Fields]  — 직접 사용 가능한 데이터")
    print("─" * 70)
    for i, f in enumerate(result.get("primary_fields", []), 1):
        print(f"\n  {i}. {f['id']}")
        print(f"     카테고리  : {f.get('category', '')}")
        print(f"     설명(원문): {f.get('description', '')}")
        print(f"     추천 이유 : {f.get('reason', '')}")

    print("\n\n🔄 [Proxy Fields]  — 대용치 데이터")
    print("─" * 70)
    for i, f in enumerate(result.get("proxy_fields", []), 1):
        print(f"\n  {i}. {f['id']}")
        print(f"     카테고리  : {f.get('category', '')}")
        print(f"     설명(원문): {f.get('description', '')}")
        print(f"     대용치 이유: {f.get('reason', '')}")

    notes = result.get("notes", "")
    if notes:
        print(f"\n\n💡 [추가 조언]\n  {notes}")

    print("\n" + "═" * 70)



def save_result(result: dict, strategy: str, out_path: str = "추천결과.xlsx"):
    rows = []
    for f in result.get("primary_fields", []):
        rows.append({
            "구분"      : "Primary",
            "id"        : f["id"],
            "category"  : f.get("category", ""),
            "description": f.get("description", ""),
            "추천 이유" : f.get("reason", ""),
        })
    for f in result.get("proxy_fields", []):
        rows.append({
            "구분"      : "Proxy",
            "id"        : f["id"],
            "category"  : f.get("category", ""),
            "description": f.get("description", ""),
            "추천 이유" : f.get("reason", ""),
        })

    df_out = pd.DataFrame(rows)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        # 메타 시트
        pd.DataFrame([{
            "전략 설명"  : strategy,
            "전략 요약"  : result.get("strategy_summary", ""),
            "추가 조언"  : result.get("notes", ""),
        }]).to_excel(w, sheet_name="요약", index=False)
        # 필드 시트
        df_out.to_excel(w, sheet_name="추천 필드", index=False)

    print(f"\n💾 결과 저장 완료: {out_path}")


def main():
    print("=" * 70)
    print("  WorldQuant Brain 데이터 필드 추천 봇  (Gemini 1.5 Flash)")
    print("  종료: 'q' 또는 'exit' 입력")
    print("=" * 70)

    # 데이터 로드
    print(f"\n📂 데이터 필드 로딩 중... ({FIELDS_FILE})")
    df = load_fields(FIELDS_FILE)
    print(f"   총 {len(df)}개 필드 로드 완료")

    field_context = build_field_context(df)

    # Gemini 초기화
    print("🤖 Gemini 1.5 Flash 초기화 중...")
    model = init_gemini(GEMINI_API_KEY)
    print("   초기화 완료\n")

    while True:
        print()
        strategy = input("전략을 설명하세요 → ").strip()

        if strategy.lower() in ("q", "exit", "quit", "종료"):
            print("봇을 종료합니다.")
            break
        if not strategy:
            continue

        print("\n⏳ 분석 중...")
        try:
            result = recommend_fields(model, field_context, strategy)
            print_result(result)

            # 저장 여부
            save = input("\n결과를 Excel로 저장할까요? (y/n) → ").strip().lower()
            if save == "y":
                fname = input("파일명 (엔터 = 추천결과.xlsx) → ").strip()
                if not fname:
                    fname = "추천결과.xlsx"
                save_result(result, strategy, fname)

        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류: {e}")
            print("   모델 응답을 다시 확인하세요.")
        except Exception as e:
            print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()

# python 노승종_개인/web_bot.py