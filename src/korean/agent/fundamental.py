"""
펀더멘털 에이전트
OpenDART 재무 수치 + LangChain RAG(보고서 원문) → Claude Haiku 분석
"""
import os
import re
import sys
import json
from datetime import datetime

_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_KOREAN_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _KOREAN_DIR)

from config import MODELS

# 지연 초기화 — API 키 없어도 import 성공
_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _get_rag_context(ticker: str, stock_name: str) -> tuple:
    """RAG에서 해당 종목 보고서 청크 검색"""
    try:
        from rag.vectorstore import get_retriever
        retriever = get_retriever(ticker=ticker, source="opendart", k=3)
        fin_year = datetime.now().year - 1
        docs = retriever.invoke(
            f"{stock_name} 재무 실적 매출 영업이익 성장 전망 {fin_year}년 {fin_year - 1}년"
        )
        texts = [d.page_content for d in docs]
        return "\n\n".join(texts), texts
    except Exception as e:
        print(f"[WARN] RAG 검색 실패: {e}")
        return "", []


def _get_financial_data(ticker: str) -> dict:
    """
    최신 재무 수치 반환.
    1순위: 로컬 reports_raw.json (빠름, 네트워크 불필요)
    2순위: DART API 실시간 조회 (로컬 파일 없을 때 폴백)
    """
    # 1. 로컬 JSON 우선 — Streamlit Cloud에도 커밋된 파일
    try:
        reports_path = os.path.abspath(
            os.path.join(_KOREAN_DIR, "..", "..", "outputs", "korean", "reports", "reports_raw.json")
        )
        if os.path.exists(reports_path):
            with open(reports_path, encoding="utf-8") as f:
                reports = json.load(f)
            ticker_reports = [
                r for r in reports
                if r.get("ticker") == ticker and r.get("financial_summary")
            ]
            if ticker_reports:
                latest = sorted(ticker_reports, key=lambda x: x.get("report_date", ""), reverse=True)[0]
                fin = dict(latest["financial_summary"])
                fin["fin_year"] = latest.get("report_date", "")[:4]
                return fin
    except Exception:
        pass

    # 2. DART API 폴백 (로컬 파일이 없을 경우)
    try:
        data_dir = os.path.join(_KOREAN_DIR, "data")
        sys.path.insert(0, data_dir)
        from fetch_reports import get_corp_code, fetch_financial_summary
        corp_code = get_corp_code(ticker)
        if not corp_code:
            return {}
        for year_offset in range(1, 4):
            fin_year = str(datetime.now().year - year_offset)
            data = fetch_financial_summary(corp_code, fin_year, "11011")
            if data:
                data["fin_year"] = fin_year
                return data
        return {}
    except Exception:
        return {}


def _parse_signal(text: str) -> int:
    """'신호:' 라인에서 매수/매도 추출 — 전체 텍스트 단순 검색보다 정확"""
    for line in text.splitlines():
        if "신호" in line:
            clean = re.sub(r"[*_#\[\]:]", "", line)
            # 신호 키워드 뒤에 오는 첫 번째 단어로 판단
            after = clean.split("신호")[-1].strip()
            if "매도" in after:
                return -1
            if "매수" in after:
                return 1
    # 폴백: 키워드 빈도 비교
    buy  = text.count("매수")
    sell = text.count("매도")
    if buy > sell:
        return 1
    if sell > buy:
        return -1
    return 0


def _parse_confidence(text: str) -> float:
    """LLM 출력에서 신뢰도 값 추출 (0.0~1.0) — 백분율(80) / 소수(0.8) 모두 처리"""
    m = re.search(r'신뢰도\s*[:：]\s*([0-9]+\.?[0-9]*)', text)
    if m:
        val = float(m.group(1))
        if val > 1.0:
            val = val / 100.0
        return min(max(val, 0.0), 1.0)
    return 0.5


def fundamental_agent(ticker: str, stock_name: str) -> tuple:
    """
    Returns:
        (result_dict, rag_text_list)
    """
    fin = _get_financial_data(ticker)
    rag_text, rag_docs = _get_rag_context(ticker, stock_name)

    fin_year_label = f" ({fin.get('fin_year', '')}년)" if fin.get('fin_year') else ""
    fin_section = (
        f"매출액: {fin.get('revenue', 'N/A')}원\n"
        f"영업이익: {fin.get('operating_profit', 'N/A')}원\n"
        f"당기순이익: {fin.get('net_income', 'N/A')}원"
        f"{fin_year_label}"
    ) if fin else "재무 데이터 없음"

    rag_section = rag_text if rag_text else "검색된 보고서 없음"

    prompt = f"""당신은 한국 주식 펀더멘털 분석 전문가입니다.

[재무 수치 (OpenDART)]
{fin_section}

[관련 보고서 발췌 (RAG 검색 — 실제 공시 문서)]
{rag_section}

위 실제 문서와 수치를 근거로 {stock_name}({ticker})의 펀더멘털을 분석하세요.
형식:
신호: 매수 또는 매도
신뢰도: 0.0~1.0
근거: 2문장 이내 (문서 인용 포함)"""

    try:
        resp = _get_client().messages.create(
            model=MODELS["claude"],
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        signal = _parse_signal(text)
        return {
            "signal":     signal,
            "confidence": _parse_confidence(text),
            "raw_output": text,
            "summary":    text.split("근거:")[-1].strip()[:120] if "근거:" in text else text[:120],
        }, rag_docs
    except Exception as e:
        return {"signal": 0, "confidence": 0.0, "raw_output": f"[ERROR] {e}", "summary": "분석 실패"}, []
