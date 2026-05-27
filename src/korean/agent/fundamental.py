"""
펀더멘털 에이전트
OpenDART 재무 수치 + LangChain RAG(보고서 원문) → GPT-4o-mini 분석
"""
import os
import re
import sys
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
        from openai import OpenAI
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        _client = OpenAI(api_key=api_key)
    return _client


def _get_rag_context(ticker: str, stock_name: str) -> tuple:
    """RAG에서 해당 종목 보고서 청크 검색"""
    try:
        from rag.vectorstore import get_retriever
        retriever = get_retriever(ticker=ticker, source="opendart", k=3)
        docs = retriever.invoke(f"{stock_name} 재무 실적 매출 영업이익 성장 전망")
        texts = [d.page_content for d in docs]
        return "\n\n".join(texts), texts
    except Exception as e:
        print(f"[WARN] RAG 검색 실패: {e}")
        return "", []


def _get_financial_data(ticker: str) -> dict:
    """OpenDART 최신 재무 수치 조회 (최근 사업보고서 기준)"""
    try:
        data_dir = os.path.join(_KOREAN_DIR, "data")
        sys.path.insert(0, data_dir)
        from fetch_reports import get_corp_code, fetch_financial_summary
        corp_code = get_corp_code(ticker)
        if not corp_code:
            return {}
        fin_year = str(datetime.now().year - 1)
        return fetch_financial_summary(corp_code, fin_year, "11011")
    except Exception:
        return {}


def _parse_confidence(text: str) -> float:
    """LLM 출력에서 신뢰도 값 추출 (0.0~1.0)"""
    m = re.search(r'신뢰도\s*[:：]\s*([0-9]+\.?[0-9]*)', text)
    if m:
        return min(max(float(m.group(1)), 0.0), 1.0)
    return 0.5


def fundamental_agent(ticker: str, stock_name: str) -> tuple:
    """
    Returns:
        (result_dict, rag_text_list)
    """
    fin = _get_financial_data(ticker)
    rag_text, rag_docs = _get_rag_context(ticker, stock_name)

    fin_section = (
        f"매출액: {fin.get('revenue', 'N/A')}원\n"
        f"영업이익: {fin.get('operating_profit', 'N/A')}원\n"
        f"당기순이익: {fin.get('net_income', 'N/A')}원"
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
        resp = _get_client().chat.completions.create(
            model=MODELS["openai"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
        signal = 1 if "매수" in text else (-1 if "매도" in text else 0)
        return {
            "signal":     signal,
            "confidence": _parse_confidence(text),
            "raw_output": text,
            "summary":    text.split("근거:")[-1].strip()[:120] if "근거:" in text else text[:120],
        }, rag_docs
    except Exception as e:
        return {"signal": 0, "confidence": 0.0, "raw_output": f"[ERROR] {e}", "summary": "분석 실패"}, []
