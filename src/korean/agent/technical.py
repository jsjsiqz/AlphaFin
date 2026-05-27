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
    ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20
    current = close.iloc[-1]

    _macd_last   = macd.iloc[-1]
    _sig_last    = signal_line.iloc[-1]
    golden_cross = (pd.notna(_macd_last) and pd.notna(_sig_last)
                    and float(_macd_last) > float(_sig_last))
    above_ma20   = pd.notna(ma20) and current > ma20
    above_ma60   = pd.notna(ma60) and current > ma60

    # 긍정 신호 카운트
    bullish = sum([golden_cross, above_ma20, above_ma60, rsi < 70])
    bearish = sum([not golden_cross, not above_ma20, rsi > 70])

    if bullish >= 3:
        signal, conf = 1, min(0.5 + bullish * 0.1, 0.95)
    elif bearish >= 2:
        signal, conf = -1, min(0.5 + bearish * 0.1, 0.90)
    else:
        signal, conf = 0, 0.4

    detail = {
        "current_price": int(current),
        "ma5":           round(float(ma5), 0),
        "ma20":          round(float(ma20), 0),
        "macd_cross":    "골든크로스" if golden_cross else "데드크로스",
        "rsi":           rsi,
        "rsi_status":    "과매수" if rsi > 70 else ("과매도" if rsi < 30 else "중립"),
        "above_ma20":    above_ma20,
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
