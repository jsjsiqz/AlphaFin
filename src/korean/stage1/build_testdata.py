"""
보고서 + 뉴스 + 주가 라벨을 결합하여 LLM 추론용 testdata 생성
AlphaFin의 korean_testdata.json 생성 (llm_inference.py 입력)
"""
import os
import sys
import json
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import OUTPUT_DIR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))
from fetch_prices import get_label

INSTRUCTION = (
    "당신은 한국 주식시장 전문 애널리스트입니다. "
    "아래 기업의 재무 공시 정보와 관련 뉴스를 분석하여 "
    "보고서 발표 다음 달의 주가 방향을 예측하세요. "
    "반드시 '상승' 또는 '하락' 중 하나로만 답하고, "
    "그 근거를 2~3문장으로 설명하세요."
)


def load_json(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_news_index(news_list: list) -> dict:
    """ticker → [news, ...] 인덱스 생성"""
    index: dict[str, list] = {}
    for item in news_list:
        ticker = item.get("ticker", "")
        index.setdefault(ticker, []).append(item)
    return index


def format_financial(fin: dict) -> str:
    def fmt(v):
        try:
            return f"{int(v):,}원"
        except Exception:
            return str(v) if v else "정보없음"

    lines = []
    if "revenue" in fin:
        lines.append(f"  매출액: {fmt(fin['revenue'])}")
    if "operating_profit" in fin:
        lines.append(f"  영업이익: {fmt(fin['operating_profit'])}")
    if "net_income" in fin:
        lines.append(f"  당기순이익: {fmt(fin['net_income'])}")
    return "\n".join(lines) if lines else "  재무 데이터 없음"


def format_news(news_list: list, max_count: int = 5) -> str:
    if not news_list:
        return "  관련 뉴스 없음"
    lines = []
    for item in news_list[:max_count]:
        title = item.get("title", "").strip()
        desc  = item.get("description", "").strip()
        date  = item.get("pub_date", "")[:16]
        lines.append(f"  [{date}] {title} - {desc[:80]}")
    return "\n".join(lines)


def build_input_text(report: dict, news_index: dict) -> str:
    ticker     = report["ticker"]
    name       = report["stock_name"]
    rpt_date   = report["report_date"]
    rpt_type   = report["report_type"]
    fin        = report.get("financial_summary", {})
    news_items = news_index.get(ticker, [])

    fin_text  = format_financial(fin)
    news_text = format_news(news_items)

    return (
        f"[기업 정보]\n"
        f"  종목명: {name} ({ticker})\n"
        f"  보고서 종류: {rpt_type}\n"
        f"  보고서 제출일: {rpt_date}\n\n"
        f"[재무 요약]\n{fin_text}\n\n"
        f"[최근 뉴스]\n{news_text}"
    )


def build_testdata(
    reports_path: str = None,
    news_path:    str = None,
    save_path:    str = None,
) -> list:
    if reports_path is None:
        reports_path = os.path.join(OUTPUT_DIR, "reports", "reports_raw.json")
    if news_path is None:
        news_path    = os.path.join(OUTPUT_DIR, "news", "news_raw.json")
    if save_path is None:
        save_path    = os.path.join(OUTPUT_DIR, "korean_testdata.json")

    print(f"[INFO] 보고서 로드: {reports_path}")
    reports  = load_json(reports_path)
    print(f"[INFO] 뉴스 로드:   {news_path}")
    news_list = load_json(news_path)

    news_index = build_news_index(news_list)
    testdata   = []
    skipped    = 0

    for report in tqdm(reports, desc="testdata 생성"):
        ticker     = report["ticker"]
        stock_name = report["stock_name"]
        rpt_date   = report["report_date"]

        label = get_label(ticker, rpt_date)
        if label == 0:
            skipped += 1
            continue

        input_text  = build_input_text(report, news_index)
        output_text = "상승" if label == 1 else "하락"

        testdata.append({
            "ticker":     ticker,
            "stock_name": stock_name,
            "date":       rpt_date,
            "instruction": INSTRUCTION,
            "input":      input_text,
            "output":     output_text,
            "label":      label,
        })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(testdata, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] {len(testdata)}건 testdata 저장 (스킵: {skipped}건): {save_path}")
    return testdata


if __name__ == "__main__":
    build_testdata()
