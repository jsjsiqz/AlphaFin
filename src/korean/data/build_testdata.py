"""
AlphaFin stage1_testdata.json의 한국판 구축
보고서 + 주가 라벨을 결합하여 LLM 추론용 데이터셋 생성
"""
import os
import sys
import json
from tqdm import tqdm

# 실행 위치에 관계없이 임포트 가능하도록 경로 이중 등록
_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_KOREAN_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _KOREAN_DIR)
sys.path.insert(0, _THIS_DIR)

from config import OUTPUT_DIR
from fetch_prices import get_label
from fetch_reports import build_report_db


INSTRUCTION = (
    "다음은 한국 상장기업의 공시 보고서 요약입니다. "
    "제시된 재무 정보와 보고서 내용을 분석하여 "
    "해당 보고서 발표 후 다음 달의 주가 방향을 예측하세요. "
    "반드시 '상승' 또는 '하락' 중 하나를 포함하여 근거와 함께 답변하세요."
)


def format_input(entry: dict) -> str:
    """
    AlphaFin의 ChatGLM 프롬프트 형식을 한국어로 재현
    """
    fin = entry.get("financial_summary", {})
    revenue = fin.get("revenue", "정보 없음")
    op_profit = fin.get("operating_profit", "정보 없음")
    net_income = fin.get("net_income", "정보 없음")

    return (
        f"이것은 {entry['stock_name']}({entry['ticker']})의 "
        f"{entry['report_date']}에 발표된 {entry['report_type']}입니다.\n\n"
        f"[재무 요약]\n"
        f"- 매출액: {revenue}원\n"
        f"- 영업이익: {op_profit}원\n"
        f"- 당기순이익: {net_income}원\n\n"
        f"위 정보를 바탕으로 향후 1개월 주가 방향을 예측하세요."
    )


def build_testdata(reports: list[dict] = None) -> list[dict]:
    """
    reports가 없으면 새로 수집, 있으면 라벨만 붙여서 반환

    출력 형식 (AlphaFin testdata 동일):
    {
        "instruction": "...",
        "input": "...",
        "output": "이 종목은 상승할 것으로 예상됩니다.",  ← ground truth
        "ticker": "005930",
        "stock_name": "삼성전자",
        "date": "2023-03-31"
    }
    """
    if reports is None:
        raw_path = os.path.join(OUTPUT_DIR, "reports", "reports_raw.json")
        if os.path.exists(raw_path):
            with open(raw_path, encoding="utf-8") as f:
                reports = json.load(f)
            print(f"[INFO] 기존 보고서 로드: {len(reports)}건")
        else:
            print("[INFO] 보고서 새로 수집 중...")
            reports = build_report_db()

    testdata = []
    skipped = 0

    for entry in tqdm(reports, desc="라벨 생성"):
        label = get_label(entry["ticker"], entry["report_date"])
        if label == 0:
            skipped += 1
            continue

        ground_truth = "상승" if label == 1 else "하락"
        output_text = (
            f"분석 결과, 이 종목은 보고서 발표 후 다음 달에 {ground_truth}할 것으로 예상됩니다."
        )

        testdata.append({
            "instruction": INSTRUCTION,
            "input": format_input(entry),
            "output": output_text,
            "ticker": entry["ticker"],
            "stock_name": entry["stock_name"],
            "date": entry["report_date"],
            "label": label,
        })

    save_dir = os.path.join(OUTPUT_DIR)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "korean_testdata.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(testdata, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] testdata {len(testdata)}건 저장 (스킵: {skipped}건)")
    print(f"상승: {sum(1 for d in testdata if d['label'] == 1)}건 / "
          f"하락: {sum(1 for d in testdata if d['label'] == -1)}건")
    return testdata


if __name__ == "__main__":
    build_testdata()
