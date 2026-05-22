"""
수집된 데이터 품질 검증
build_testdata.py 실행 후 반드시 확인하세요
"""
import json
import os
import sys
from collections import Counter
import pandas as pd

_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_KOREAN_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _KOREAN_DIR)

from config import OUTPUT_DIR, TARGET_STOCKS


def verify():
    testdata_path = os.path.join(OUTPUT_DIR, "korean_testdata.json")

    if not os.path.exists(testdata_path):
        print("❌ korean_testdata.json 없음 → build_testdata.py 먼저 실행")
        return

    with open(testdata_path, encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 55)
    print(f" 총 데이터 건수: {len(data)}건")
    print("=" * 55)

    # 1. 라벨 분포
    labels = [d["label"] for d in data]
    up   = labels.count(1)
    down = labels.count(-1)
    print(f"\n[라벨 분포]")
    print(f"  상승(1):  {up}건  ({up/len(data)*100:.1f}%)")
    print(f"  하락(-1): {down}건  ({down/len(data)*100:.1f}%)")
    if abs(up - down) / len(data) > 0.3:
        print(f"  ⚠️  불균형 주의 (발표 시 명시 필요)")

    # 2. 종목별 데이터 건수
    ticker_counts = Counter(d["ticker"] for d in data)
    missing = [name for ticker, name in TARGET_STOCKS.items()
               if ticker not in ticker_counts]
    print(f"\n[종목 커버리지]")
    print(f"  수집된 종목: {len(ticker_counts)}/30개")
    if missing:
        print(f"  ⚠️  데이터 없는 종목: {', '.join(missing)}")

    # 3. 기간 분포
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    print(f"\n[기간 분포]")
    print(f"  시작: {df['date'].min().date()}")
    print(f"  종료: {df['date'].max().date()}")

    by_year = df.groupby(df["date"].dt.year).size()
    for year, cnt in by_year.items():
        print(f"  {year}년: {cnt}건")

    # 4. 재무 데이터 품질
    missing_fin = sum(
        1 for d in data
        if not d["input"].strip() or "정보 없음" in d["input"]
    )
    print(f"\n[재무 데이터 품질]")
    print(f"  정상: {len(data) - missing_fin}건")
    print(f"  재무정보 없음: {missing_fin}건")

    # 5. 샘플 1건 출력
    print(f"\n[샘플 데이터 (1건)]")
    sample = data[0]
    print(f"  종목: {sample['stock_name']} ({sample['ticker']})")
    print(f"  날짜: {sample['date']}")
    print(f"  라벨: {'상승' if sample['label']==1 else '하락'}")
    print(f"  입력:\n{sample['input'][:200]}...")

    print("\n" + "=" * 55)
    if len(data) >= 100 and missing_fin / len(data) < 0.2:
        print("✅ 데이터 품질 양호 → llm_inference.py 실행 가능")
    else:
        print("⚠️  데이터 품질 확인 필요")


if __name__ == "__main__":
    verify()
