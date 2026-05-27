"""
감성 에이전트
LangChain RAG(뉴스) + Stage 1 예측값 → GPT-4o-mini 감성 분류
"""
import os
import re
import sys
import pandas as pd

_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_KOREAN_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _KOREAN_DIR)

from config import OPENAI_API_KEY, MODELS, OUTPUT_DIR

# 지연 초기화 — API 키 없어도 import 성공
_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _get_news_context(ticker: str, stock_name: str) -> tuple:
    """RAG에서 해당 종목 뉴스 청크 검색"""
    try:
        from rag.vectorstore import get_retriever
        retriever = get_retriever(ticker=ticker, source="naver_news", k=5)
        docs = retriever.invoke(f"{stock_name} 주가 뉴스 공시 실적 전망")
        texts = [d.page_content for d in docs]
        return "\n\n".join(texts), texts
    except Exception as e:
        print(f"[WARN] 뉴스 RAG 검색 실패: {e}")
        return "", []


def _get_stage1_signal(ticker: str) -> dict:
    """Stage 1 parsed_predictions.xlsx에서 해당 종목 최신 예측 조회"""
    pred_path = os.path.join(OUTPUT_DIR, "parsed_predictions.xlsx")
    if not os.path.exists(pred_path):
        return {}
    try:
        df = pd.read_excel(pred_path)
        # Excel이 '005930' 같은 종목코드를 정수(5930)로 읽는 문제 방지
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        df_t = df[df["ticker"] == ticker].sort_values("date", ascending=False)
        if df_t.empty:
            return {}
        latest     = df_t.iloc[0]
        model_cols = [c for c in ["claude", "openai"] if c in df.columns]
        if not model_cols:
            return {}
        signals   = {m: int(latest.get(m, 0)) for m in model_cols}
        consensus = 1 if sum(signals.values()) > 0 else -1
        agreement = f"{sum(v == consensus for v in signals.values())}/{len(signals)}"
        return {
            "signals":     signals,
            "consensus":   consensus,
            "agreement":   agreement,
            "report_date": str(latest.get("date", ""))[:10],
        }
    except Exception:
        return {}


def _parse_confidence(text: str) -> float:
    """LLM 출력에서 신뢰도 값 추출 (0.0~1.0)"""
    m = re.search(r'신뢰도\s*[:：]\s*([0-9]+\.?[0-9]*)', text)
    if m:
        return min(max(float(m.group(1)), 0.0), 1.0)
    return 0.5


def sentiment_agent(ticker: str, stock_name: str) -> tuple:
    """
    Returns:
        (result_dict, rag_text_list)
    """
    news_text, news_docs = _get_news_context(ticker, stock_name)
    stage1 = _get_stage1_signal(ticker)

    news_section = news_text if news_text else "수집된 뉴스 없음"

    stage1_section = (
        f"LLM 합의: {stage1.get('agreement','N/A')} 일치 / "
        f"기준 보고서: {stage1.get('report_date','N/A')}"
    ) if stage1 else "Stage 1 데이터 없음 (Stage 1 먼저 실행 필요)"

    prompt = f"""당신은 한국 주식 시장 감성 분석 전문가입니다.

[최신 뉴스 (RAG 검색 — 실제 수집 뉴스)]
{news_section}

[Stage 1 멀티 LLM 예측 참고]
{stage1_section}

위 뉴스와 예측을 종합하여 {stock_name}({ticker})의 시장 감성을 평가하세요.
형식:
신호: 긍정 또는 부정
신뢰도: 0.0~1.0
핵심 뉴스: 가장 중요한 뉴스 한 줄
근거: 2문장 이내"""

    try:
        resp = _get_client().chat.completions.create(
            model=MODELS["openai"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
        signal = 1 if "긍정" in text else (-1 if "부정" in text else 0)
        return {
            "signal":      signal,
            "confidence":  _parse_confidence(text),
            "stage1_data": stage1,
            "raw_output":  text,
            "summary":     text.split("근거:")[-1].strip()[:120] if "근거:" in text else text[:120],
        }, news_docs
    except Exception as e:
        return {"signal": 0, "confidence": 0.0, "raw_output": f"[ERROR] {e}", "summary": "분석 실패"}, []
