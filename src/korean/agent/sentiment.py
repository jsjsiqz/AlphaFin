"""
감성 에이전트
LangChain RAG(뉴스) + Stage 1 예측값 → Claude Haiku 감성 분류
"""
import os
import re
import sys
import pandas as pd
from datetime import datetime

_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_KOREAN_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _KOREAN_DIR)

from config import MODELS, OUTPUT_DIR

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


def _get_news_context(ticker: str, stock_name: str) -> tuple:
    """RAG에서 해당 종목 뉴스 청크 검색"""
    try:
        from rag.vectorstore import get_retriever
        retriever = get_retriever(ticker=ticker, source="naver_news", k=5)
        cur_year = datetime.now().year
        docs = retriever.invoke(
            f"{stock_name} 주가 뉴스 공시 실적 전망 {cur_year}년 {cur_year - 1}년"
        )
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
        s = sum(signals.values())
        consensus = 1 if s > 0 else (-1 if s < 0 else 0)
        agreement = f"{sum(v == consensus for v in signals.values())}/{len(signals)}"
        return {
            "signals":     signals,
            "consensus":   consensus,
            "agreement":   agreement,
            "report_date": str(latest.get("date", ""))[:10],
        }
    except Exception:
        return {}


def _parse_signal(text: str) -> int:
    """'신호:' 라인에서 긍정/부정 추출 — 전체 텍스트 단순 검색보다 정확"""
    for line in text.splitlines():
        if "신호" in line:
            clean = re.sub(r"[*_#\[\]:]", "", line)
            # 신호 키워드 뒤에 오는 첫 번째 단어로 판단
            after = clean.split("신호")[-1].strip()
            if "부정" in after:
                return -1
            if "긍정" in after:
                return 1
            if "중립" in after:
                return 0
    # 폴백: 키워드 빈도 비교
    pos = text.count("긍정")
    neg = text.count("부정")
    if pos > neg:
        return 1
    if neg > pos:
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


def sentiment_agent(ticker: str, stock_name: str) -> tuple:
    """
    Returns:
        (result_dict, rag_text_list)
    """
    news_text, news_docs = _get_news_context(ticker, stock_name)
    stage1 = _get_stage1_signal(ticker)

    news_section = news_text if news_text else "수집된 뉴스 없음"

    if stage1:
        _sig_map = {1: "상승(매수)", -1: "하락(매도)", 0: "중립"}
        _sig_lines = ", ".join(
            f"{m}: {_sig_map.get(v, '?')}"
            for m, v in stage1.get("signals", {}).items()
        )
        _consensus_kr = _sig_map.get(stage1.get("consensus", 0), "?")
        stage1_section = (
            f"개별 신호: {_sig_lines}\n"
            f"합의 방향: {_consensus_kr} ({stage1.get('agreement','N/A')} 일치)\n"
            f"기준 보고서: {stage1.get('report_date','N/A')}"
        )
    else:
        stage1_section = "Stage 1 데이터 없음 (Stage 1 먼저 실행 필요)"

    prompt = f"""당신은 한국 주식 시장 감성 분석 전문가입니다.

[최신 뉴스 (RAG 검색 — 실제 수집 뉴스)]
{news_section}

[Stage 1 멀티 LLM 예측 참고]
{stage1_section}

위 뉴스와 예측을 종합하여 {stock_name}({ticker})의 시장 감성을 평가하세요.
형식:
신호: 긍정, 중립, 부정 중 하나
신뢰도: 0.0~1.0
핵심 뉴스: 가장 중요한 뉴스 한 줄
근거: 2문장 이내"""

    try:
        resp = _get_client().messages.create(
            model=MODELS["claude"],
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        signal = _parse_signal(text)
        return {
            "signal":      signal,
            "confidence":  _parse_confidence(text),
            "stage1_data": stage1,
            "raw_output":  text,
            "summary":     text.split("근거:")[-1].strip()[:120] if "근거:" in text else text[:120],
        }, news_docs
    except Exception as e:
        return {"signal": 0, "confidence": 0.0, "raw_output": f"[ERROR] {e}", "summary": "분석 실패"}, []
