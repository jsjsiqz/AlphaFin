"""
pykrx를 이용한 한국 주가 데이터 수집
AlphaFin의 Tushare 기반 수집을 대체
"""
import pandas as pd
from pykrx import stock
from tqdm import tqdm
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import TARGET_STOCKS, BACKTEST_START, BACKTEST_END, OUTPUT_DIR


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """일별 OHLCV + 시가총액 수집"""
    df = stock.get_market_ohlcv(start, end, ticker)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    try:
        df_cap = stock.get_market_cap(start, end, ticker)
        if not df_cap.empty and "시가총액" in df_cap.columns:
            df["market_cap"] = df_cap["시가총액"]
        else:
            df["market_cap"] = 0
    except Exception:
        df["market_cap"] = 0
    return df


def fetch_monthly_return(ticker: str, start: str, end: str) -> pd.Series:
    """월말 종가 기준 월별 수익률"""
    df = fetch_ohlcv(ticker, start, end)
    if df.empty or "종가" not in df.columns:
        return pd.Series(dtype=float)
    monthly = df["종가"].resample("M").last()
    return monthly.pct_change()


def build_price_db(save_dir: str = None) -> dict:
    """
    전체 종목의 주가 데이터를 수집하여 딕셔너리로 반환
    Returns: {ticker: DataFrame}
    """
    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, "prices")
    os.makedirs(save_dir, exist_ok=True)

    price_db = {}
    for ticker, name in tqdm(TARGET_STOCKS.items(), desc="주가 수집"):
        try:
            df = fetch_ohlcv(ticker, BACKTEST_START, BACKTEST_END)
            price_db[ticker] = df
            df.to_csv(os.path.join(save_dir, f"{ticker}_{name}.csv"), encoding="utf-8-sig")
        except Exception as e:
            print(f"[ERROR] {name}({ticker}): {e}")

    print(f"\n[완료] {len(price_db)}개 종목 저장: {save_dir}")
    return price_db


def get_label(ticker: str, report_date: str) -> int:
    """
    보고서 발표 다음 달 수익률로 상승(1)/하락(-1) 라벨 생성
    AlphaFin의 ground truth 방식과 동일

    예시: 2023-03-31 보고서 → 4월 수익률 (3월말 대비 4월말)
    """
    try:
        report_dt = pd.to_datetime(report_date)
        start = report_dt.strftime("%Y%m%d")
        # 2달치 데이터를 가져와야 다음달 수익률(pct_change)이 계산됨
        end = (report_dt + pd.offsets.MonthEnd(2)).strftime("%Y%m%d")

        monthly = fetch_monthly_return(ticker, start, end)
        # dropna() 후 첫 번째 값 = 다음 달(보고서 발표 월+1) 수익률
        monthly = monthly.dropna()
        if len(monthly) < 1:
            return 0
        target = monthly.iloc[0]
        return 1 if target > 0 else -1
    except Exception as e:
        print(f"[WARN] 라벨 생성 실패 {ticker} {report_date}: {e}")
        return 0


if __name__ == "__main__":
    build_price_db()
