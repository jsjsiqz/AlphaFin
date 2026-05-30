"""
멀티에이전트 오케스트레이터
순수 Python 함수 파이프라인 — LangGraph 없이 동일한 3에이전트 구조 구현

흐름: technical → fundamental → sentiment → synthesizer
각 에이전트는 독립 함수이며 AgentState를 통해 결과를 공유합니다.
"""
import os
import sys

_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_KOREAN_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _KOREAN_DIR)
sys.path.insert(0, _THIS_DIR)

from state import AgentState
from technical   import technical_agent
from fundamental import fundamental_agent
from sentiment   import sentiment_agent
from synthesizer import synthesizer


def run(ticker: str) -> AgentState:
    """
    멀티에이전트 파이프라인 실행 진입점
    n8n, Streamlit, CLI 모두 이 함수 하나로 호출
    """
    from config import TARGET_STOCKS
    from pykrx import stock as pykrx_stock
    import pandas as pd

    state = AgentState(
        ticker=ticker,
        stock_name=TARGET_STOCKS.get(ticker, ticker),
    )

    # 현재가 조회 — 주말/공휴일이면 최근 영업일로 소급
    for _delta in range(0, 5):
        _date = (pd.Timestamp.today() - pd.Timedelta(days=_delta)).strftime("%Y%m%d")
        _df   = pykrx_stock.get_market_ohlcv(_date, _date, ticker)
        if not _df.empty and "종가" in _df.columns:
            state.current_price = float(_df["종가"].iloc[-1])
            break

    # ── 에이전트 순차 실행 ────────────────────────────────────────
    state.tech_result = technical_agent(ticker)

    fund_result, fund_docs = fundamental_agent(ticker, state.stock_name)
    state.fund_result = fund_result
    state.rag_context.extend(fund_docs)

    sent_result, sent_docs = sentiment_agent(ticker, state.stock_name)
    state.sent_result = sent_result
    state.rag_context.extend(sent_docs)

    # ── 합성 ─────────────────────────────────────────────────────
    # synthesizer는 dict를 받으므로 dataclass를 dict로 변환
    state.recommendation = synthesizer(state.__dict__)

    # ── 다수결 합의 신호 ──────────────────────────────────────────
    signals = [
        (state.tech_result  or {}).get("signal", 0),
        (state.fund_result  or {}).get("signal", 0),
        (state.sent_result  or {}).get("signal", 0),
    ]
    valid = [s for s in signals if s != 0]
    s = sum(valid)
    state.final_signal = 1 if s > 0 else (-1 if s < 0 else 0)

    return state


if __name__ == "__main__":
    import json as _json
    import argparse

    parser = argparse.ArgumentParser(description="AlphaFin 에이전트 파이프라인")
    parser.add_argument("--ticker", default="005930", help="종목 티커 (예: 005930)")
    parser.add_argument(
        "--output", choices=["text", "json"], default="text",
        help="출력 형식 — json: n8n Execute Command 노드용"
    )
    args = parser.parse_args()

    result = run(args.ticker)

    if args.output == "json":
        # n8n Execute Command 노드가 stdout을 JSON으로 파싱
        print(_json.dumps({
            "ticker":         result.ticker,
            "stock_name":     result.stock_name,
            "final_signal":   result.final_signal,
            "recommendation": result.recommendation,
            "tech_signal":    (result.tech_result  or {}).get("signal", 0),
            "fund_signal":    (result.fund_result  or {}).get("signal", 0),
            "sent_signal":    (result.sent_result  or {}).get("signal", 0),
            "tech_summary":   (result.tech_result  or {}).get("summary", ""),
            "fund_summary":   (result.fund_result  or {}).get("summary", ""),
            "sent_summary":   (result.sent_result  or {}).get("summary", ""),
        }, ensure_ascii=False))
    else:
        signal_kr = {1: "매수", -1: "매도", 0: "중립"}
        print(f"\n[최종 결과] {result.stock_name}({result.ticker})")
        print(f"신호: {signal_kr.get(result.final_signal, '중립')}")
        print(f"\n{result.recommendation}")
