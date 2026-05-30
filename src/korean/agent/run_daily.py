"""
n8n Execute Command 노드용 — 일일 전 종목 에이전트 분석
평일 09:10 자동 실행 → signal==1 종목만 Telegram 알림
"""
import sys
import os
import json
import time
import argparse

_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_KOREAN_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _KOREAN_DIR)
sys.path.insert(0, _THIS_DIR)

from config import TARGET_STOCKS
from graph import run as run_agent


def main():
    parser = argparse.ArgumentParser(description="AlphaFin 일일 전 종목 에이전트 분석")
    parser.add_argument("--output", choices=["text", "json"], default="text")
    parser.add_argument(
        "--tickers", nargs="*", default=None,
        help="분석 대상 티커 (기본: TARGET_STOCKS 전체 30종목)",
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="종목 간 대기 초 — API 레이트 리밋 방지 (기본: 2.0)",
    )
    args = parser.parse_args()

    tickers = args.tickers or list(TARGET_STOCKS.keys())

    results    = []
    buy_signals = []
    errors     = []

    for i, ticker in enumerate(tickers):
        stock_name = TARGET_STOCKS.get(ticker, ticker)
        if args.output == "text":
            print(f"[{i+1}/{len(tickers)}] {stock_name}({ticker}) 분석 중...", flush=True)

        try:
            state = run_agent(ticker)
            item = {
                "ticker":         ticker,
                "stock_name":     stock_name,
                "final_signal":   state.final_signal,
                "tech_signal":    (state.tech_result  or {}).get("signal", 0),
                "fund_signal":    (state.fund_result  or {}).get("signal", 0),
                "sent_signal":    (state.sent_result  or {}).get("signal", 0),
                "recommendation": (state.recommendation or "")[:300],
            }
            results.append(item)
            if state.final_signal == 1:
                buy_signals.append(item)
        except Exception as e:
            errors.append({"ticker": ticker, "stock_name": stock_name, "error": str(e)})
            print(f"[ERROR] {stock_name}({ticker}): {e}", file=sys.stderr, flush=True)

        if i < len(tickers) - 1:
            time.sleep(args.delay)

    if args.output == "json":
        print(json.dumps({
            "total":       len(results),
            "buy_count":   len(buy_signals),
            "error_count": len(errors),
            "buy_signals": buy_signals,
        }, ensure_ascii=False))
    else:
        sig_map = {1: "🟢 매수", -1: "🔴 매도", 0: "🟡 중립"}
        print("\n── 분석 결과 ──────────────────────────────")
        for r in results:
            print(f"  {r['stock_name']}({r['ticker']}): {sig_map[r['final_signal']]}")
        if buy_signals:
            print(f"\n매수 신호 {len(buy_signals)}종목: {[s['stock_name'] for s in buy_signals]}")
        if errors:
            print(f"오류 {len(errors)}건: {[e['ticker'] for e in errors]}", file=sys.stderr)


if __name__ == "__main__":
    main()
