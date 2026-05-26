"""
한국 주식 롱숏 전략 백테스트 (AlphaFin test_strategy.py의 한국판)
pykrx 기반으로 Tushare 완전 대체

[데이터 수집 전략]
- 월별 종가: fetch_prices.py가 생성한 캐시(prices/monthly_close.csv) 우선 사용
- 벤치마크:  30종목 동일가중 수익률 (KRX 지수 API 인증 불필요)
- 시가총액:  KRX 인증 필요로 사용 안 함 → 동일가중 전략 적용
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fire import Fire
from pykrx import stock as pykrx_stock

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import OUTPUT_DIR

PRICES_CACHE_DIR = os.path.join(OUTPUT_DIR, "prices")

# 한글 폰트 설정 (OS별 자동 선택)
import platform
_os = platform.system()
if _os == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif _os == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# ── 캐시 로더 ─────────────────────────────────────────────────────────────

def load_cached_monthly_close() -> pd.DataFrame:
    """fetch_prices.py 가 생성한 월별 종가 캐시 로드"""
    path = os.path.join(PRICES_CACHE_DIR, "monthly_close.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    df.index = pd.to_datetime(df.index)
    return df


def load_cached_benchmark() -> pd.Series | None:
    """fetch_prices.py 가 생성한 벤치마크 캐시 로드"""
    path = os.path.join(PRICES_CACHE_DIR, "benchmark_monthly.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    df.index = pd.to_datetime(df.index)
    return df.iloc[:, 0]


# ── 실시간 수집 폴백 (캐시 없을 때만) ─────────────────────────────────────

def fetch_monthly_close(tickers: list, start: str, end: str) -> pd.DataFrame:
    """월말 종가 DataFrame 반환 — 캐시 없을 때 실시간 수집"""
    frames = {}
    for ticker in tqdm(tickers, desc="주가 실시간 수집"):
        try:
            df = pykrx_stock.get_market_ohlcv(start, end, ticker)
            if df.empty or "종가" not in df.columns:
                continue
            df.index = pd.to_datetime(df.index)
            frames[ticker] = df["종가"].resample("ME").last()
        except Exception as e:
            print(f"[WARN] {ticker}: {e}")
    return pd.DataFrame(frames)


def build_benchmark_from_close(df_close: pd.DataFrame) -> pd.Series:
    """
    30종목 동일가중 수익률 → KOSPI 대체 벤치마크.
    KRX 지수 API 인증 불필요.
    """
    ret = df_close.pct_change()
    bm = ret.mean(axis=1).rename("KOSPI_proxy(동일가중30)")
    return bm


# ── 성과 지표 ─────────────────────────────────────────────────────────────

def calc_metrics(rr: pd.Series, benchmark: pd.Series = None) -> dict:
    """AlphaFin의 get_지표()와 동일한 지표 계산"""
    months = len(rr)
    total_ret = rr.sum()
    ann_ret = total_ret / (months / 12) if months > 0 else 0
    ann_vol = rr.std() * (12 ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + rr).cumprod()
    mdd = ((cum / cum.cummax()) - 1).min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0

    metrics = {
        "총수익률(%)":      round(total_ret * 100, 2),
        "연환산수익률(%)":  round(ann_ret * 100, 2),
        "연환산변동성(%)":  round(ann_vol * 100, 2),
        "샤프비율":         round(sharpe, 3),
        "최대낙폭(%)":      round(mdd * 100, 2),
        "카르마비율":       round(calmar, 3),
    }

    if benchmark is not None:
        bm = benchmark.reindex(rr.index).fillna(0)
        bm_ann = bm.sum() / (months / 12) if months > 0 else 0
        metrics["벤치마크연환산(%)"] = round(bm_ann * 100, 2)
        metrics["초과수익률(%)"]    = round((ann_ret - bm_ann) * 100, 2)

    return metrics


# ── 백테스트 메인 ─────────────────────────────────────────────────────────

def run_backtest(
    pred_path: str = f"{OUTPUT_DIR}/parsed_predictions.xlsx",
    save_dir:  str = f"{OUTPUT_DIR}/backtest",
    weight:    str = "동일가중",   # KRX 시총 API 인증 불필요 → 동일가중 고정
    long_short: str = "롱숏",
):
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_excel(pred_path)
    df["date"] = pd.to_datetime(df["date"])
    # Excel이 '005930' 같은 종목코드를 정수(5930)로 읽는 문제 방지
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["next_month"] = (
        df["date"].dt.to_period("M")
        .apply(lambda p: (p + 1).to_timestamp("M"))
    )

    model_cols = [
        c for c in df.columns
        if c not in ("ticker", "stock_name", "date", "next_month", "ground_truth")
    ]

    # ── 1. 월별 종가 로드 ─────────────────────────────────────────────────
    df_close = load_cached_monthly_close()
    if df_close.empty:
        print("[INFO] 캐시 없음 → 실시간 수집 중... (fetch_prices.py 먼저 실행 권장)")
        tickers = df["ticker"].unique().tolist()
        start = df["date"].min().strftime("%Y%m%d")
        end   = df["next_month"].max().strftime("%Y%m%d")
        df_close = fetch_monthly_close(tickers, start, end)
    else:
        print(f"[INFO] 월별 종가 캐시 로드: {df_close.shape[1]}종목 × {df_close.shape[0]}개월")

    if df_close.empty:
        raise RuntimeError("주가 데이터를 수집할 수 없습니다. fetch_prices.py를 먼저 실행하세요.")

    # ── 2. 벤치마크 로드 ──────────────────────────────────────────────────
    benchmark = load_cached_benchmark()
    if benchmark is None:
        print("[INFO] 벤치마크 캐시 없음 → 30종목 동일가중으로 계산")
        benchmark = build_benchmark_from_close(df_close)
    else:
        print("[INFO] 벤치마크 캐시 로드 완료")

    # ── 3. 수익률 계산 ────────────────────────────────────────────────────
    df_ret = df_close.pct_change()

    if weight == "시총가중":
        print("[INFO] KRX 시총 API 인증 필요 → 동일가중으로 대체")

    # ── 4. 모델별 포트폴리오 구성 ─────────────────────────────────────────
    ports = {}
    for model in tqdm(model_cols, desc="모델별 백테스트"):
        pivot = df.pivot_table(
            index="next_month", columns="ticker", values=model, aggfunc="first"
        ).fillna(0)

        ret    = df_ret.reindex(index=pivot.index, columns=pivot.columns).fillna(0)
        signal = (
            pivot * (pivot > 0) if long_short == "롱only"  else
            pivot * (pivot < 0) if long_short == "숏only"  else
            pivot
        )

        denom    = signal.abs().sum(axis=1).replace(0, np.nan)
        port_ret = (ret * signal).sum(axis=1) / denom
        ports[model] = port_ret.fillna(0)

    df_ports = pd.DataFrame(ports)

    # ── 5. 성과 지표 계산 ─────────────────────────────────────────────────
    metrics_list = {}
    for model in model_cols:
        metrics_list[model] = calc_metrics(df_ports[model], benchmark)
    bm_label = benchmark.name or "KOSPI_proxy"
    metrics_list[bm_label] = calc_metrics(benchmark.reindex(df_ports.index).fillna(0))

    df_metrics = pd.DataFrame(metrics_list).T
    print("\n[성과 지표 비교]")
    print(df_metrics.to_string())

    # ── 6. 결과 저장 ──────────────────────────────────────────────────────
    metrics_path = os.path.join(save_dir, f"metrics_{weight}_{long_short}.csv")
    df_metrics.to_csv(metrics_path, encoding="utf-8-sig")

    # ── 7. 누적 수익률 차트 ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10.colors

    for i, model in enumerate(model_cols):
        cum = (1 + df_ports[model]).cumprod() - 1
        ax.plot(cum.index, cum.values * 100, label=model, color=colors[i % len(colors)], linewidth=2)

    bm_aligned = benchmark.reindex(df_ports.index).fillna(0)
    bm_cum = (1 + bm_aligned).cumprod() - 1
    ax.plot(bm_cum.index, bm_cum.values * 100, label=bm_label, color="gray", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Accumulated Return (%)", fontsize=12)
    ax.set_title(f"전략별 누적 수익률 비교 ({weight}, {long_short})", fontsize=13)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    chart_path = os.path.join(save_dir, f"returns_{weight}_{long_short}.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[완료] 차트 저장: {chart_path}")

    return df_metrics


def main(
    pred_path:  str = f"{OUTPUT_DIR}/parsed_predictions.xlsx",
    save_dir:   str = f"{OUTPUT_DIR}/backtest",
    weight:     str = "동일가중",
    long_short: str = "롱숏",
):
    run_backtest(pred_path, save_dir, weight, long_short)


if __name__ == "__main__":
    Fire(main)
