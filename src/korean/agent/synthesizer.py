"""
합성 에이전트
3개 에이전트 결과 + RAG 컨텍스트 → GPT-4o-mini → 자연어 투자 의견
"""
import os
import sys

_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_KOREAN_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _KOREAN_DIR)

from config import MODELS

# 지연 초기화 — API 키 없어도 import 성공
_client = None

_SIGNAL_KR = {1: "매수", -1: "매도", 0: "중립"}


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        _client = OpenAI(api_key=api_key)
    return _client


def synthesizer(state: dict) -> str:
    """
    AgentState를 받아 자연어 투자 의견 생성

    Args:
        state: AgentState (tech_result, fund_result, sent_result, rag_context 포함)
    Returns:
        자연어 추천 문자열
    """
    ticker     = state.get("ticker", "")
    stock_name = state.get("stock_name", "")
    price      = state.get("current_price", 0)
    tech       = state.get("tech_result") or {}
    fund       = state.get("fund_result") or {}
    sent       = state.get("sent_result") or {}
    rag_ctx    = state.get("rag_context", [])

    tech_signal = _SIGNAL_KR.get(tech.get("signal", 0), "중립")
    fund_signal = _SIGNAL_KR.get(fund.get("signal", 0), "중립")
    sent_signal = "긍정" if sent.get("signal", 0) > 0 else "부정"

    # RAG 컨텍스트 요약 (상위 2개 청크 — 문자열 리스트)
    rag_summary = "\n".join(str(c) for c in rag_ctx[:2]) if rag_ctx else "추가 문서 없음"

    stage1 = sent.get("stage1_data", {})
    stage1_info = (
        f"Stage 1 LLM 합의: {stage1.get('agreement','N/A')}, "
        f"기준일: {stage1.get('report_date','N/A')}"
    ) if stage1 else "Stage 1 데이터 없음"

    prompt = f"""
당신은 한국 주식 투자 전문 AI 어드바이저입니다.
아래 3개 전문 에이전트의 분석 결과와 실제 공시·뉴스 문서를 종합하여
{stock_name}({ticker})에 대한 투자 의견을 제시하세요.

[현재가]
{price:,}원

[기술 에이전트 분석]
신호: {tech_signal} (신뢰도: {tech.get('confidence', 0):.0%})
요약: {tech.get('summary', '-')}

[펀더멘털 에이전트 분석 — RAG 보고서 기반]
신호: {fund_signal} (신뢰도: {fund.get('confidence', 0):.0%})
요약: {fund.get('summary', '-')}

[감성 에이전트 분석 — RAG 뉴스 + Stage 1 기반]
신호: {sent_signal} (신뢰도: {sent.get('confidence', 0):.0%})
요약: {sent.get('summary', '-')}
{stage1_info}

[RAG 검색된 실제 문서 발췌]
{rag_summary}

위 분석을 종합하여 다음 형식으로 답하세요:

최종 의견: 매수 / 중립 / 매도 중 하나
핵심 근거: 3가지 불릿포인트로 각 에이전트 근거 인용
주의사항: 리스크 1가지
"""
    try:
        resp = _get_client().chat.completions.create(
            model=MODELS["openai"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] 합성 실패: {e}"
