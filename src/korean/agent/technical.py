"""
기술 에이전트
pykrx 60일 주가 데이터 → MACD / 이동평균 / RSI 분석
(RAG 불필요 — 수치 기반 결정적 분석)
"""
import pandas as pd
import numpy as np

from pykrx import stock as pykrx_stock


def _fetch_recent_price(ticker: str, days: int = 90) -> pd.Series:
    """최근 N일 종가 시계열 반환"""
    end = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y%m%d")
    df = pykrx_stock.get_market_ohlcv(start, end, ticker)
    if df.empty:
        return pd.Series(dtype=float)
    df.index = pd.to_datetime(df.index)
    return df["종가"]


def _calc_macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def _calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(period).mean()
    loss   = (-delta.clip(upper=0)).rolling(period).mean()
    rs     = gain / loss.replace(0, np.nan)
    rsi    = 100 - (100 / (1 + rs))
    if rsi.empty:
        return 50.0
    val = rsi.iloc[-1]
    return round(float(val), 1) if pd.notna(val) else 50.0


def technical_agent(ticker: str) -> dict:
    """
    Returns:
        {signal, confidence, detail, summary}
    """
    close = _fetch_recent_price(ticker)

    if len(close) < 30:
        return {
            "signal": 0, "confidence": 0.0,
            "detail": {}, "summary": "주가 데이터 부족"
        }

    macd, signal_line = _calc_macd(close)
    rsi = _calc_rsi(close)

    ma5  = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    current = close.iloc[-1]

    _macd_last   = macd.iloc[-1]
    _sig_last    = signal_line.iloc[-1]
    golden_cross = (pd.notna(_macd_last) and pd.notna(_sig_last)
                    and float(_macd_last) > float(_sig_last))
    above_ma20   = pd.notna(ma20) and current > ma20

    # ma60은 데이터가 충분할 때만 독립 신호로 활용 (부족하면 집계 제외)
    has_ma60   = len(close) >= 60
    ma60       = close.rolling(60).mean().iloc[-1] if has_ma60 else None
    above_ma60 = (pd.notna(ma60) and current > ma60) if has_ma60 else None

    # 긍정 신호 카운트
    bullish_signals = [golden_cross, above_ma20, rsi < 70]
    bearish_signals = [not golden_cross, not above_ma20, rsi > 70]
    if has_ma60:
        bullish_signals.append(above_ma60)
        bearish_signals.append(not above_ma60)
    bullish = sum(bullish_signals)
    bearish = sum(bearish_signals)

    total = len(bullish_signals)
    if bullish >= max(3, total - 1):
        signal, conf = 1, min(0.5 + bullish * 0.1, 0.95)
    elif bearish >= max(2, total - 1):
        signal, conf = -1, min(0.5 + bearish * 0.1, 0.90)
    else:
        signal, conf = 0, 0.4

    detail = {
        "current_price": int(current),
        "ma5":           round(float(ma5), 0),
        "ma20":          round(float(ma20), 0),
        "ma60":          round(float(ma60), 0) if has_ma60 and pd.notna(ma60) else None,
        "macd_cross":    "골든크로스" if golden_cross else "데드크로스",
        "rsi":           rsi,
        "rsi_status":    "과매수" if rsi > 70 else ("과매도" if rsi < 30 else "중립"),
        "above_ma20":    above_ma20,
        "above_ma60":    above_ma60,
    }

    summary = (
        f"{'골든크로스' if golden_cross else '데드크로스'}, "
        f"RSI {rsi}({detail['rsi_status']}), "
        f"20일선 {'위' if above_ma20 else '아래'}"
    )

    return {
        "signal":     signal,
        "confidence": round(conf, 2),
        "detail":     detail,
        "summary":    summary,
    }
