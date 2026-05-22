#!/bin/bash
# AlphaFin Korean — 전체 파이프라인 실행
# 사용법: bash src/korean/run_pipeline.sh
#
# 실행 순서:
#   Step 0: 연결 테스트
#   Step 1: 보고서 수집 (OpenDART)
#   Step 2: 뉴스 수집 (네이버)
#   Step 3: 테스트 데이터 구축
#   Step 4: 멀티 LLM 추론
#   Step 5: 정확도 계산
#   Step 6: 백테스트
#   Step 7: RAG 인덱스 구축
#   (에이전트 데모는 streamlit run app.py 로 별도 실행)

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
echo "[Step 3] 테스트 데이터 구축..."
python data/build_testdata.py

echo ""
echo "[Step 3-1] 수집 품질 확인..."
python data/verify_data.py

echo ""
echo "[Step 4] 멀티 LLM 추론 (~21분)..."
python stage1/llm_inference.py \
    --models gemini groq openai \
    --data_path ../../outputs/korean/korean_testdata.json \
    --output_dir ../../outputs/korean

echo ""
echo "[Step 5] 정확도 계산..."
python stage1/postprocess.py \
    --pred_path ../../outputs/korean/llm_predictions.jsonl \
    --save_path ../../outputs/korean/parsed_predictions.xlsx

echo ""
echo "[Step 6] 롱숏 전략 백테스트..."
python stage1/backtest.py \
    --pred_path ../../outputs/korean/parsed_predictions.xlsx \
    --save_dir ../../outputs/korean/backtest \
    --weight 시총가중 \
    --long_short 롱숏

echo ""
echo "[Step 7] RAG 인덱스 구축 (~3분, OpenAI 임베딩 ~\$0.01)..."
python rag/indexer.py

echo ""
echo "========================================"
echo "   파이프라인 완료!                      "
echo "   에이전트 데모 실행 (현재 디렉토리):   "
echo "   streamlit run app.py                 "
echo "========================================"
