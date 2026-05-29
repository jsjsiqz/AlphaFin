"""
pykrx를 이용한 한국 주가 데이터 수집
AlphaFin의 Tushare 기반 수집을 대체

[KRX API 인증 안내]
- get_market_ohlcv : 인증 불필요 ✅ → 주가 수집에 사용
- get_market_cap   : KRX_ID/KRX_PW 필요 ❌ → 사용 안 함 (동일가중 대체)
- get_index_ohlcv  : KRX_ID/KRX_PW 필요 ❌ → 사용 안 함 (30종목 동일가중 프록시 사용)
"""
import os
import sys
import time
import pandas as pd
from pykrx import stock
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import TARGET_STOCKS, BACKTEST_START, BACKTEST_END, OUTPUT_DIR

PRICES_CACHE_DIR = os.path.join(OUTPUT_DIR, "prices")


# ── 재시도 래퍼 ────────────────────────────────────────────────────────────

def _retry_pykrx(func, *args, max_retries: int = 3, delay: float = 3.0, **kwargs):
    """
    pykrx API 호출 실패 시 지수 백오프로 재시도.
    모든 재시도 실패 시 None 반환 (크래시 방지).
    """
    last_exc = None
    for i in range(max_retries):
        try:
            result = func(*args, **kwargs)
            if result is not None and not (hasattr(result, "empty") and result.empty):
                return result
        except Exception as e:
            last_exc = e
            if i < max_retries - 1:
                wait = delay * (i + 1)
                print(f"  [재시도 {i+1}/{max_retries}] {func.__name__}: {e} — {wait:.0f}초 대기")
                time.sleep(wait)
    print(f"  [WARN] {func.__name__} 최종 실패: {last_exc}")
    return None


# ── 개별 종목 OHLCV 수집 ──────────────────────────────────────────────────

def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    일별 OHLCV 수집 (재시도 3회).
    시가총액은 KRX 인증 필요로 수집하지 않음.
    """
    df = _retry_pykrx(stock.get_market_ohlcv, start, end, ticker)
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    return df


def fetch_monthly_return(ticker: str, start: str, end: str) -> pd.Series:
    """월말 종가 기준 월별 수익률"""
    df = fetch_ohlcv(ticker, start, end)
    if df.empty or "종가" not in df.columns:
        return pd.Series(dtype=float)
    monthly = df["종가"].resample("ME").last()
    return monthly.pct_change()


# ── 월별 종가 캐시 구축 ────────────────────────────────────────────────────

def build_monthly_close_cache(save_dir: str = None) -> pd.DataFrame:
    """
    전체 30종목의 월말 종가를 수집하여 CSV로 저장.
    backtest.py가 이 파일을 읽어 API 재호출 없이 동작.

    저장 파일:
        prices/monthly_close.csv    — 종목별 월말 종가
        prices/benchmark_monthly.csv — 30종목 동일가중 수익률 (KOSPI 프록시)
    """
    if save_dir is None:
        save_dir = PRICES_CACHE_DIR
    os.makedirs(save_dir, exist_ok=True)

    frames = {}
    for ticker, name in tqdm(TARGET_STOCKS.items(), desc="월별 종가 수집"):
        try:
            df = fetch_ohlcv(ticker, BACKTEST_START, BACKTEST_END)
            if df.empty or "종가" not in df.columns:
                print(f"  [SKIP] {name}({ticker}): 데이터 없음")
                continue
            frames[ticker] = df["종가"].resample("ME").last()
        except Exception as e:
            print(f"  [ERROR] {name}({ticker}): {e}")

    df_close = pd.DataFrame(frames)
    close_path = os.path.join(save_dir, "monthly_close.csv")
    df_close.to_csv(close_path, encoding="utf-8-sig")
    print(f"\n[완료] 월별 종가 저장: {close_path}")
    print(f"       {len(df_close)}개월 × {len(df_close.columns)}종목")

    # 동일가중 30종목 수익률 → KOSPI 대체 벤치마크
    bm = df_close.pct_change().mean(axis=1).rename("KOSPI_proxy")
    bm_path = os.path.join(save_dir, "benchmark_monthly.csv")
    bm.to_csv(bm_path, encoding="utf-8-sig")
    print(f"[완료] 벤치마크(30종목 동일가중) 저장: {bm_path}")

    return df_close


# ── 기존 호환 함수 ─────────────────────────────────────────────────────────

def build_price_db(save_dir: str = None) -> dict:
    """
    전체 종목의 일별 OHLCV를 종목별 CSV로 저장.
    동시에 build_monthly_close_cache()도 실행하여 백테스트용 캐시 생성.
    """
    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, "prices")
    os.makedirs(save_dir, exist_ok=True)

    price_db = {}
    for ticker, name in tqdm(TARGET_STOCKS.items(), desc="일별 주가 수집"):
        try:
            df = fetch_ohlcv(ticker, BACKTEST_START, BACKTEST_END)
            price_db[ticker] = df
            df.to_csv(os.path.join(save_dir, f"{ticker}_{name}.csv"), encoding="utf-8-sig")
        except Exception as e:
            print(f"[ERROR] {name}({ticker}): {e}")

    print(f"\n[완료] {len(price_db)}개 종목 일별 데이터 저장: {save_dir}")

    # 백테스트용 월별 캐시도 함께 생성
    build_monthly_close_cache(save_dir)

    return price_db


def get_label(ticker: str, report_date: str) -> int:
    """
    보고서 발표 다음 달 수익률로 상승(1)/하락(-1) 라벨 생성.
    캐시가 있으면 캐시에서 읽고, 없으면 실시간 수집.
    """
    # 캐시 우선 조회
    close_path = os.path.join(PRICES_CACHE_DIR, "monthly_close.csv")
    if os.path.exists(close_path):
        try:
            df_close = pd.read_csv(close_path, index_col=0, encoding="utf-8-sig")
            df_close.index = pd.to_datetime(df_close.index)
            if ticker in df_close.columns:
                report_dt = pd.to_datetime(report_date)
                # Period 산술로 "다음 달 말일" 계산 — MonthEnd(1)은 월 중순 날짜를 당월 말로 롤업하므로 오프바이원 발생
                target_month = (pd.Period(report_date, 'M') + 1).to_timestamp('M')
                col = df_close[ticker].dropna()
                # 보고서 발표월 말 → 다음 달 말 수익률
                prev = col[col.index <= report_dt.replace(day=1) + pd.offsets.MonthEnd(0)]
                nxt  = col[col.index <= target_month]
                if len(prev) >= 1 and len(nxt) >= 1:
                    ret = (nxt.iloc[-1] - prev.iloc[-1]) / prev.iloc[-1]
                    return 1 if ret > 0 else -1
        except Exception:
            pass

    # 캐시 없으면 실시간 수집
    try:
        report_dt = pd.to_datetime(report_date)
        start = report_dt.strftime("%Y%m%d")
        end = (report_dt + pd.offsets.MonthEnd(2)).strftime("%Y%m%d")
        monthly = fetch_monthly_return(ticker, start, end).dropna()
        if len(monthly) < 1:
            return 0
        return 1 if monthly.iloc[0] > 0 else -1
    except Exception as e:
        print(f"[WARN] 라벨 생성 실패 {ticker} {report_date}: {e}")
        return 0


if __name__ == "__main__":
    build_price_db()
