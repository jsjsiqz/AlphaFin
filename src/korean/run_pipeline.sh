#!/bin/bash
# AlphaFin Korean — 전체 파이프라인 실행
# 사용법: bash src/korean/run_pipeline.sh
#
# 실행 순서:
#   Step 0: 연결 테스트
#   Step 1: 보고서 수집 (OpenDART)
#   Step 2: 뉴스 수집 (네이버)
#   Step 3: 주가 수집 + 월별 캐시 구축 (pykrx OHLCV)
#   Step 4: 테스트 데이터 구축
#   Step 5: 수집 품질 확인
#   Step 6: 멀티 LLM 추론
#   Step 7: 정확도 계산
#   Step 8: 롱숏 전략 백테스트 (캐시에서 읽음, API 재호출 없음)
#   Step 9: RAG 인덱스 구축
#   (에이전트 데모는 streamlit run app.py 로 별도 실행)
#
# [KRX API 인증 안내]
#   get_market_ohlcv  → 인증 불필요, Step 3에서 사용
#   get_market_cap    → KRX_ID/KRX_PW 필요, 사용 안 함 (동일가중 대체)
#   get_index_ohlcv   → KRX_ID/KRX_PW 필요, 사용 안 함 (30종목 프록시 대체)

set -e
cd "$(dirname "$0")"

echo ""
echo "========================================"
echo "   AlphaFin Korean 파이프라인 시작       "
echo "========================================"

echo ""
echo "[Step 0] API 연결 테스트..."
python data/test_connection.py

echo ""
echo "[Step 1] OpenDART 보고서 수집 (~20분)..."
python data/fetch_reports.py

echo ""
echo "[Step 2] 네이버 뉴스 수집 (~5분)..."
python data/fetch_news.py

echo ""
echo "[Step 3] 주가 수집 + 월별 캐시 구축 (~5분)..."
echo "         outputs/korean/prices/monthly_close.csv    생성"
echo "         outputs/korean/prices/benchmark_monthly.csv 생성"
python data/fetch_prices.py

echo ""
echo "[Step 4] 테스트 데이터 구축..."
python data/build_testdata.py

echo ""
echo "[Step 5] 수집 품질 확인..."
python data/verify_data.py

echo ""
echo "[Step 6] 멀티 LLM 추론 (~10분)..."
python stage1/llm_inference.py \
    --models claude openai \
    --data_path ../../outputs/korean/korean_testdata.json \
    --output_dir ../../outputs/korean

echo ""
echo "[Step 7] 정확도 계산..."
python stage1/postprocess.py \
    --pred_path ../../outputs/korean/llm_predictions.jsonl \
    --save_path ../../outputs/korean/parsed_predictions.xlsx

echo ""
echo "[Step 8] 롱숏 전략 백테스트 (캐시 기반, KRX API 재호출 없음)..."
python stage1/backtest.py \
    --pred_path ../../outputs/korean/parsed_predictions.xlsx \
    --save_dir ../../outputs/korean/backtest \
    --weight 동일가중 \
    --long_short 롱숏

echo ""
echo "[Step 9] RAG 인덱스 구축 (~3분, OpenAI 임베딩 ~\$0.01)..."
python rag/indexer.py

echo ""
echo "========================================"
echo "   파이프라인 완료!                      "
echo "   에이전트 데모 실행 (현재 디렉토리):   "
echo "   streamlit run app.py                 "
echo "========================================"
