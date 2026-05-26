"""
LLM 출력에서 상승/하락 추출 (AlphaFin dataprocess_stockgpt.py의 한국판)
Claude(Haiku) / OpenAI GPT-4o-mini 한국어 금융 키워드 기반 규칙 추출
"""
import json
import os
import sys
import pandas as pd
from fire import Fire

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import OUTPUT_DIR

# 한국어 금융 키워드 (AlphaFin의 중국어 키워드 대체)
KEYWORDS_UP = [
    "상승", "오를", "올라", "상향", "매수", "긍정", "호조", "개선",
    "성장", "강세", "돌파", "반등", "회복", "기대", "유망", "추천",
]
KEYWORDS_DOWN = [
    "하락", "내릴", "내려", "하향", "매도", "부정", "부진", "악화",
    "감소", "약세", "조정", "리스크", "우려", "불안", "주의", "경계",
]

# LLM 출력에서 결론 구분자 (AlphaFin의 "因此"/"综上所述" 대응)
CONCLUSION_MARKERS = ["따라서", "결론적으로", "종합하면", "분석 결과", "예상됩니다", "전망됩니다"]


def extract_direction(text: str) -> int:
    """
    LLM 출력 텍스트에서 방향 추출
    Returns: 1(상승), -1(하락), 0(판단 불가)
    AlphaFin의 getPredUpDownStrict() + getPredUpDownLoose() 통합
    """
    if not text or text.startswith("[ERROR]"):
        return 0

    # 결론 구분자 이후 텍스트에서 우선 탐색
    for marker in CONCLUSION_MARKERS:
        if marker in text:
            idx = text.find(marker)
            conclusion = text[idx:]
            for kw in KEYWORDS_UP:
                if kw in conclusion:
                    return 1
            for kw in KEYWORDS_DOWN:
                if kw in conclusion:
                    return -1

    # 결론 구분자 없으면 전체 텍스트에서 탐색
    for kw in KEYWORDS_UP:
        if kw in text:
            return 1
    for kw in KEYWORDS_DOWN:
        if kw in text:
            return -1

    return 0


def extract_ground_truth(text: str) -> int:
    """ground_truth 텍스트에서 라벨 추출"""
    if "상승" in text:
        return 1
    elif "하락" in text:
        return -1
    return 0


def process_predictions(
    pred_path: str,
    save_path: str = None,
) -> pd.DataFrame:
    """
    llm_predictions.jsonl을 읽어 방향값으로 변환 후 DataFrame 반환
    AlphaFin의 main()에서 df_all 만드는 과정과 동일
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "parsed_predictions.xlsx")

    records = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    model_names = [
        k for k in records[0].keys()
        if k not in ("ticker", "stock_name", "date", "ground_truth", "label")
    ]

    rows = []
    for r in records:
        row = {
            # zfill(6): Excel이 '005930'을 정수(5930)로 저장하는 문제 방지
            "ticker": str(r["ticker"]).zfill(6),
            "stock_name": r["stock_name"],
            "date": r["date"],
            "ground_truth": r.get("label", extract_ground_truth(r["ground_truth"])),
        }
        for model in model_names:
            row[model] = extract_direction(r.get(model, ""))
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 판단 불가(0) 비율 출력
    for model in model_names:
        zero_pct = (df[model] == 0).mean() * 100
        print(f"[{model}] 판단 불가 비율: {zero_pct:.1f}%")

    # 판단 불가는 다수결(앙상블)로 대체 (AlphaFin 방식)
    for idx, row in df.iterrows():
        for model in model_names:
            if row[model] == 0:
                others = [row[m] for m in model_names if m != model and row[m] != 0]
                if others:
                    df.at[idx, model] = 1 if sum(others) > 0 else -1

    df.to_excel(save_path, index=False)
    print(f"\n[완료] 파싱 완료: {save_path}")
    return df


def calculate_accuracy(df: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    """AlphaFin의 calculate_accuracy()와 동일"""
    results = {}
    for model in model_names:
        filtered = df[df[model] != 0]
        if len(filtered) == 0:
            results[model] = 0.0
            continue
        correct = (filtered["ground_truth"] == filtered[model]).sum()
        results[model] = round(correct / len(filtered) * 100, 2)

    acc_df = pd.DataFrame.from_dict(results, orient="index", columns=["accuracy(%)"])
    print("\n[정확도 비교]")
    print(acc_df.to_string())
    return acc_df


def main(
    pred_path: str = f"{OUTPUT_DIR}/llm_predictions.jsonl",
    save_path: str = f"{OUTPUT_DIR}/parsed_predictions.xlsx",
):
    df = process_predictions(pred_path, save_path)
    model_names = [
        c for c in df.columns
        if c not in ("ticker", "stock_name", "date", "ground_truth")
    ]
    calculate_accuracy(df, model_names)


if __name__ == "__main__":
    Fire(main)
