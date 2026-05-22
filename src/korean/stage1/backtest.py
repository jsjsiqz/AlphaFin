"""
한국 주식 롱숏 전략 백테스트 (AlphaFin test_strategy.py의 한국판)
pykrx 기반으로 Tushare 완전 대체
"""
import os
import sys
import json
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


# ── 주가 / 시가총액 수집 ───────────────────────────────────────────────────

def fetch_monthly_close(tickers: list, start: str, end: str) -> pd.DataFrame:
    """월말 종가 DataFrame 반환 (index=월, columns=ticker)"""
    frames = {}
    for ticker in tqdm(tickers, desc="주가 수집"):
        try:
            df = pykrx_stock.get_market_ohlcv(start, end, ticker)
            if df.empty or "종가" not in df.columns:
                continue
            df.index = pd.to_datetime(df.index)
            monthly = df["종가"].resample("M").last()
            frames[ticker] = monthly
        except Exception as e:
            print(f"[WARN] {ticker}: {e}")
    return pd.DataFrame(frames)


def fetch_monthly_cap(tickers: list, start: str, end: str) -> pd.DataFrame:
    """월말 시가총액 DataFrame 반환"""
    frames = {}
    for ticker in tqdm(tickers, desc="시총 수집"):
        try:
            df = pykrx_stock.get_market_cap(start, end, ticker)
            if df.empty or "시가총액" not in df.columns:
                continue
            df.index = pd.to_datetime(df.index)
            monthly = df["시가총액"].resample("M").last()
            frames[ticker] = monthly
        except Exception:
            pass
    return pd.DataFrame(frames)


def fetch_benchmark(start: str, end: str) -> pd.Series:
    """KOSPI 지수 월별 수익률 (AlphaFin의 CSI300 대체)"""
    df = pykrx_stock.get_index_ohlcv(start, end, "1001")  # 1001 = KOSPI
    df.index = pd.to_datetime(df.index)
    monthly = df["종가"].resample("M").last()
    return monthly.pct_change().rename("KOSPI")


# ── 성과 지표 ─────────────────────────────────────────────────────────────

def calc_metrics(rr: pd.Series, benchmark: pd.Series = None) -> dict:
    """
    AlphaFin의 get_지표()와 동일한 지표 계산
    - 연환산수익률(ARR), 샤프비율, 최대낙폭(MDD), 카르마비율
    """
    months = len(rr)
    total_ret = rr.sum()
    ann_ret = total_ret / (months / 12)
    ann_vol = rr.std() * (12 ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + rr).cumprod()
    mdd = ((cum / cum.cummax()) - 1).min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0

    metrics = {
        "총수익률(%)": round(total_ret * 100, 2),
        "연환산수익률(%)": round(ann_ret * 100, 2),
        "연환산변동성(%)": round(ann_vol * 100, 2),
        "샤프비율": round(sharpe, 3),
        "최대낙폭(%)": round(mdd * 100, 2),
        "카르마비율": round(calmar, 3),
    }

    if benchmark is not None:
        bm = benchmark.reindex(rr.index).fillna(0)
        bm_ann = bm.sum() / (months / 12)
        metrics["벤치마크연환산(%)"] = round(bm_ann * 100, 2)
        metrics["초과수익률(%)"] = round((ann_ret - bm_ann) * 100, 2)

    return metrics


# ── 백테스트 메인 ─────────────────────────────────────────────────────────

def run_backtest(
    pred_path: str = f"{OUTPUT_DIR}/parsed_predictions.xlsx",
    save_dir: str = f"{OUTPUT_DIR}/backtest",
    weight: str = "시총가중",  # "시총가중" or "동일가중"
    long_short: str = "롱숏",  # "롱숏", "롱only", "숏only"
):
    """
    AlphaFin test_strategy.py의 main()과 동일한 역할
    """
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_excel(pred_path)
    df["date"] = pd.to_datetime(df["date"])
    # 보고서 발표 월의 다음 달 말일 = 전략 실행 기준 날짜
    # Period 기반 계산으로 중간 날짜(예: 5월 15일) → 6월 말 정확히 처리
    df["next_month"] = (
        df["date"].dt.to_period("M")
        .apply(lambda p: (p + 1).to_timestamp("M"))
    )

    model_cols = [
        c for c in df.columns
        if c not in ("ticker", "stock_name", "date", "next_month", "ground_truth")
    ]
    tickers = df["ticker"].unique().tolist()

    # 주가 / 시총 수집
    start = df["date"].min().strftime("%Y%m%d")
    end = df["next_month"].max().strftime("%Y%m%d")

    print("\n주가 및 시가총액 수집 중...")
    df_close = fetch_monthly_close(tickers, start, end)
    df_cap = fetch_monthly_cap(tickers, start, end)
    benchmark = fetch_benchmark(start, end)

    df_ret = df_close.pct_change()

    ports = {}
    for model in tqdm(model_cols, desc="모델별 백테스트"):
        pivot = df.pivot_table(
            index="next_month", columns="ticker", values=model, aggfunc="first"
        ).fillna(0)

        ret = df_ret.reindex(index=pivot.index, columns=pivot.columns).fillna(0)
        cap = df_cap.reindex(index=pivot.index, columns=pivot.columns).fillna(0)

        if long_short == "롱only":
            signal = pivot * (pivot > 0)
        elif long_short == "숏only":
            signal = pivot * (pivot < 0)
        else:
            signal = pivot

        if weight == "시총가중":
            weighted_cap = cap * signal.abs()
            port_ret = (ret * signal * weighted_cap).sum(axis=1) / weighted_cap.sum(axis=1).replace(0, np.nan)
        else:
            port_ret = (ret * signal).sum(axis=1) / signal.abs().sum(axis=1).replace(0, np.nan)

        ports[model] = port_ret.fillna(0)

    df_ports = pd.DataFrame(ports)

    # 성과 지표 계산
    metrics_list = {}
    for model in model_cols:
        metrics_list[model] = calc_metrics(df_ports[model], benchmark)
    metrics_list["KOSPI"] = calc_metrics(benchmark.reindex(df_ports.index).fillna(0))

    df_metrics = pd.DataFrame(metrics_list).T
    print("\n[성과 지표 비교]")
    print(df_metrics.to_string())

    # 저장
    df_metrics.to_csv(os.path.join(save_dir, f"metrics_{weight}_{long_short}.csv"), encoding="utf-8-sig")

    # 누적 수익률 차트
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10.colors

    for i, model in enumerate(model_cols):
        cum = (1 + df_ports[model]).cumprod() - 1
        ax.plot(cum.index, cum.values * 100, label=model, color=colors[i % len(colors)], linewidth=2)

    # KOSPI 벤치마크
    bm_aligned = benchmark.reindex(df_ports.index).fillna(0)
    bm_cum = (1 + bm_aligned).cumprod() - 1
    ax.plot(bm_cum.index, bm_cum.values * 100, label="KOSPI", color="gray", linestyle="--", linewidth=1.5)

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
    pred_path: str = f"{OUTPUT_DIR}/parsed_predictions.xlsx",
    save_dir: str = f"{OUTPUT_DIR}/backtest",
    weight: str = "시총가중",
    long_short: str = "롱숏",
):
    run_backtest(pred_path, save_dir, weight, long_short)


if __name__ == "__main__":
    Fire(main)
