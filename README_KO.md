# AlphaFin-Korean: RAG + 멀티에이전트 한국 주식 분석 시스템

> AlphaFin (LREC-COLING 2024) Stock-Chain 방법론을 한국 금융 시장에 적용·확장  
> 숭실대학교 정보과학대학원 AI학과 금융AI 팀프로젝트  
> 정영욱 · 오장일 · 박성현 · 지유정 · 오금환 · 이승진 · 김민우

---

## 면책 조항

본 저장소의 모든 내용은 **학술 연구 및 교육 목적으로만** 제공됩니다.  
출력되는 주가 예측 및 투자 의견은 실제 금융·법률·투자 조언이 아닙니다.

---

## 개요

원본 AlphaFin은 중국 시장 기반의 단일 LLM 파인튜닝 시스템입니다.  
본 프로젝트는 이를 네 가지 방향으로 확장합니다.

| 확장 | 내용 |
|---|---|
| **한국 시장 적용** | Tushare → pykrx, ChatGLM → Claude/GPT-4o-mini |
| **LangChain RAG** | 공시 보고서·뉴스를 Chroma 벡터DB에 인덱싱, 에이전트가 실시간 검색 |
| **멀티에이전트 파이프라인** | 기술·펀더멘털·감성 에이전트를 순수 Python 함수로 조율 |
| **n8n 자동화** | 데이터 수집·RAG 갱신·Telegram 알림을 스케줄로 자동 실행 |

### 비용 정책

| 구성요소 | 비용 |
|---|---|
| OpenDART, 네이버 API, pykrx | **모두 무료** |
| LangChain, Chroma, Streamlit, n8n | **모두 무료** |
| Claude Haiku (Anthropic) | **학교 제공** |
| OpenAI (임베딩 + 합성기 전용) | **~$0.05 / 프로젝트 전체** |

---

## 기술 스택 선택 근거

```
[선택]  LangChain     — RAG 핵심 (Document 로더·청킹·Retriever). 대체 불가.
[선택]  Chroma        — 로컬 벡터DB. 서버·비용 없음. 인덱싱 3분이면 구축.
[선택]  n8n           — 스케줄 자동화 + Telegram 알림. 이미 보유 → 추가 비용 없음.
[선택]  Claude(Haiku) — Stage 1 추론 + 에이전트 sub-분석 (학교 제공).
[선택]  GPT-4o-mini   — 최종 합성기 + 임베딩만 사용. 품질·비용 최적점.
[제외]  LangGraph     — 고정 선형 파이프라인에 불필요한 복잡도.
                        순수 Python 함수로 동일한 멀티에이전트 구현.
[제외]  Supabase      — Chroma와 중복. 팀 공유 필요 시에만 선택 확장.
[제외]  plotly        — matplotlib + Streamlit 기본 차트로 충분.
```

---

## 원본 AlphaFin과의 비교

| 구성요소 | 원본 AlphaFin (중국) | 본 프로젝트 (한국) |
|---|---|---|
| 주가 데이터 | Tushare API (유료) | **pykrx (무료)** |
| 재무 리포트 | 중국 증권사 리포트 | **OpenDART 공시 보고서** |
| LLM | ChatGLM2-6B + LoRA 파인튜닝 | **Claude(Haiku) / GPT-4o-mini API 비교** |
| 지식베이스 | FAISS (정적 샘플 200건) | **LangChain + Chroma (동적, 실수집)** |
| 에이전트 | 없음 | **3에이전트 + 합성기 파이프라인** |
| 자동화 | 없음 | **n8n 스케줄 + Telegram 알림** |
| 출력 | 상승/하락 수치 | **자연어 투자 의견 + 참조 문서 표시** |
| 벤치마크 | CSI300 | **KOSPI** |

---

## 전체 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│   LAYER 0   n8n 자동화 레이어                                    │
│                                                                  │
│  [Cron: 평일 08:50]             [Cron: 평일 09:10]              │
│  ┌────────────────────────┐   ┌────────────────────────────┐   │
│  │ 데이터 파이프라인       │   │ 에이전트 분석 + 알림        │   │
│  │ fetch_reports.py       │   │ python agent/graph.py      │   │
│  │ fetch_news.py          │   │   --ticker 005930          │   │
│  │ rag/indexer.py         │   │   --output json            │   │
│  │     ↓                  │   │         ↓                  │   │
│  │ Telegram: "수집 완료"  │   │ IF signal==1 → Telegram    │   │
│  └────────────────────────┘   └────────────────────────────┘   │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│   LAYER 1   Stage 1 배치 파이프라인 (정량 결과)                   │
│                                                                  │
│  OpenDART 보고서 (30종목 × 4회/년 × 2년 ≈ 240건)                │
│        ↓                                                         │
│  pykrx 다음달 수익률 → 상승(1) / 하락(-1) 라벨                  │
│        ↓                                                         │
│  멀티 LLM: Claude Haiku(학교제공) / GPT-4o-mini                             │
│        ↓                                                         │
│  정확도 비교표 + KOSPI 대비 롱숏 백테스트                         │
│        │                                                         │
│  reports_raw.json ──────────────────────────────┐               │
│  llm_predictions.jsonl ──────────────────────┐  │               │
└──────────────────────────────────────────────┼┼──┼──────────────┘
                                               ││  │ 재활용
┌──────────────────────────────────────────────▼▼──▼──────────────┐
│   LAYER 2   LangChain RAG 지식베이스 (Chroma 로컬)               │
│                                                                  │
│  OpenDART 보고서 청크 (source: "opendart")                       │
│  네이버 뉴스 청크     (source: "naver_news")                     │
│                                                                  │
│  임베딩: OpenAI text-embedding-3-small (~$0.01 총액)            │
│  저장:   Chroma 로컬 (outputs/korean/chroma_db/)                │
│  검색:   cosine similarity + 메타데이터 필터                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │ RAG 검색
┌────────────────────────────▼─────────────────────────────────────┐
│   LAYER 3   멀티에이전트 파이프라인 (실시간 분석)                 │
│                                                                  │
│  ┌──────────────┐  ┌────────────────────┐  ┌─────────────────┐ │
│  │ 기술 에이전트 │  │ 펀더멘털 에이전트  │  │  감성 에이전트  │ │
│  │ pykrx 60일   │  │ RAG(opendart) 검색 │  │ RAG(news) 검색  │ │
│  │ MACD/MA/RSI  │  │ + OpenDART 수치    │  │ + Stage1 예측   │ │
│  │ → 신호 계산  │  │ → Claude 분석      │  │ → Claude 분류     │ │
│  │ (LLM 없음)   │  │                    │  │                 │ │
│  └──────┬───────┘  └─────────┬──────────┘  └────────┬────────┘ │
│         └──────────────────┬─┘────────────────────────┘         │
│                            ↓                                     │
│              ┌─────────────────────────┐                        │
│              │      합성 에이전트       │                        │
│              │  3신호 + RAG 컨텍스트   │                        │
│              │  → GPT-4o-mini (~$0.001)│                        │
│              │  → 자연어 투자 의견     │                        │
│              └────────────┬────────────┘                        │
└───────────────────────────┼──────────────────────────────────────┘
                            ↓
             ┌──────────────────────────────┐
             │     Streamlit 대시보드        │
             │  3에이전트 신호 카드          │
             │  RAG 참조 문서 표시 (XAI)    │
             │  자연어 투자 의견 출력        │
             └──────────────────────────────┘
```

---

## 멀티에이전트 설계

### AgentState — 공유 상태 (dataclass)

```python
@dataclass
class AgentState:
    ticker:         str
    stock_name:     str   = ""
    current_price:  float = 0.0
    tech_result:    Optional[dict] = None   # 기술 에이전트 출력
    fund_result:    Optional[dict] = None   # 펀더멘털 에이전트 출력
    sent_result:    Optional[dict] = None   # 감성 에이전트 출력
    rag_context:    list  = field(default_factory=list)  # RAG 검색 청크
    final_signal:   int   = 0               # -1 / 0 / 1
    recommendation: str   = ""              # 자연어 투자 의견
```

### 에이전트 파이프라인 (`agent/graph.py`)

```python
def run(ticker: str) -> AgentState:
    state = AgentState(ticker=ticker, ...)

    # 순차 실행 — 각 에이전트는 독립 함수
    state.tech_result = technical_agent(ticker)

    fund_result, fund_docs = fundamental_agent(ticker, state.stock_name)
    state.fund_result = fund_result
    state.rag_context.extend(fund_docs)

    sent_result, sent_docs = sentiment_agent(ticker, state.stock_name)
    state.sent_result = sent_result
    state.rag_context.extend(sent_docs)

    state.recommendation = synthesizer(state.__dict__)

    # 다수결 합의
    signals = [tech["signal"], fund["signal"], sent["signal"]]
    state.final_signal = 1 if sum(signals) > 0 else -1

    return state
```

### 에이전트별 LLM 배치

| 에이전트 | LLM | 이유 |
|---|---|---|
| 기술 에이전트 | **없음** | MACD·RSI는 수식으로 결정적 계산 |
| 펀더멘털 에이전트 | **Claude Haiku** (학교 제공) | RAG 보고서 문서 해석 |
| 감성 에이전트 | **Claude Haiku** (학교 제공) | 뉴스 감성 분류 + Stage1 참조 |
| 합성 에이전트 | **GPT-4o-mini** | 최종 출력 품질 필요 |

### Stage 1 → 감성 에이전트 연결

```
Stage 1 실행 → parsed_predictions.xlsx 생성
                         ↓
감성 에이전트: 해당 종목의 최신 LLM 예측 조회
  예) "Claude: 상승, GPT-4o-mini: 상승 (3/3 일치)"
                         ↓
RAG 뉴스 검색 결과 + Stage1 예측 → Claude에게 전달 → 감성 분류
```

---

## n8n 자동화 설계

### 역할

n8n은 Python 스크립트를 **스케줄에 따라 자동 실행**하고, 분석 결과에 따라 **Telegram 알림을 발송**합니다.

```
Python 코드 역할: 데이터 수집, RAG 구축, 에이전트 분석
n8n 역할:         위 스크립트를 언제·어떻게 실행하고 결과를 어디로 보낼지 결정
```

### 워크플로우 1: 일일 데이터 파이프라인

```
[1] Schedule Trigger  (Cron: 50 8 * * 1-5, 평일 08:50)
[2] Execute Command   python data/fetch_reports.py
[3] Execute Command   python data/fetch_news.py
[4] Execute Command   python rag/indexer.py
[5] Telegram          ✅ AlphaFin 데이터 갱신 완료
```

### 워크플로우 2: 에이전트 분석 + 매수 신호 알림

```
[1] Schedule Trigger  (Cron: 10 9 * * 1-5, 평일 09:10)

[2] Execute Command
    python agent/graph.py --ticker 005930 --output json
    → stdout: {"final_signal": 1, "stock_name": "삼성전자", ...}

[3] IF  final_signal == 1
    ↓ (True)
[4] Telegram
    🚀 매수 신호 감지!
    종목: {{ $json.stock_name }}
    {{ $json.recommendation }}
```

### n8n ↔ Python 연결 방식

```bash
# --output json 플래그로 n8n Execute Command 노드가 stdout을 파싱
python agent/graph.py --ticker 005930 --output json

# 출력 예시
{
  "ticker": "005930",
  "stock_name": "삼성전자",
  "final_signal": 1,
  "recommendation": "매수 의견 — MACD 골든크로스...",
  "tech_signal": 1,
  "fund_signal": 1,
  "sent_signal": 1
}
```

---

## 디렉토리 구조

```
AlphaFin/
├── src/
│   └── korean/
│       ├── config.py                    # API 키, 경로, 모델 설정
│       │
│       ├── data/                        # 데이터 수집 (n8n이 자동 호출)
│       │   ├── fetch_reports.py         # OpenDART 보고서 + 재무 수치
│       │   ├── fetch_prices.py          # pykrx 월별 종가 캐시 구축 (백테스트 선행 필수)
│       │   ├── fetch_news.py            # 네이버 뉴스 (RAG 소스)
│       │   ├── build_testdata.py        # Stage 1 testdata 구축
│       │   ├── test_connection.py       # API 연결 사전 확인
│       │   └── verify_data.py           # 수집 품질 검증
│       │
│       ├── stage1/                      # 배치 파이프라인
│       │   ├── llm_inference.py         # 멀티 LLM 방향 예측
│       │   ├── postprocess.py           # 키워드 파싱 + 정확도
│       │   └── backtest.py              # 롱숏 전략 백테스트
│       │
│       ├── rag/                         # LangChain RAG
│       │   ├── loader.py                # JSON → LangChain Document
│       │   ├── vectorstore.py           # Chroma 어댑터
│       │   └── indexer.py               # 인덱스 구축 실행
│       │
│       ├── agent/                       # 멀티에이전트 파이프라인
│       │   ├── state.py                 # AgentState dataclass
│       │   ├── technical.py             # 기술 에이전트 (pykrx)
│       │   ├── fundamental.py           # 펀더멘털 에이전트 (RAG + Claude)
│       │   ├── sentiment.py             # 감성 에이전트 (RAG + Stage1 + Claude)
│       │   ├── synthesizer.py           # 합성기 (GPT-4o-mini)
│       │   └── graph.py                 # 파이프라인 + n8n JSON CLI
│       │
│       ├── app.py                       # Streamlit 대시보드
│       └── run_pipeline.sh              # Stage 1 전체 실행
│
├── outputs/
│   └── korean/
│       ├── reports/reports_raw.json     # ← RAG 인덱스 소스
│       ├── news/news_raw.json           # ← RAG 인덱스 소스
│       ├── prices/
│       │   ├── monthly_close.csv        # 30종목 × 25개월 캐시 (fetch_prices.py 생성)
│       │   └── benchmark_monthly.csv   # 동일가중 벤치마크 수익률
│       ├── chroma_db/                   # Chroma 로컬 인덱스
│       ├── korean_testdata.json
│       ├── llm_predictions.jsonl        # ← 감성 에이전트 참조
│       ├── parsed_predictions.xlsx
│       └── backtest/
│
├── .env.example
├── requirements_korean.txt
└── README_KO.md
```

---

## 빠른 시작

### 1. 설치

```bash
git clone https://github.com/your-repo/AlphaFin.git
cd AlphaFin
pip install -r requirements_korean.txt
```

### 2. API 키 설정

```bash
cp .env.example .env
```

| API | 발급처 | 비용 |
|---|---|---|
| Claude (Anthropic) | 학교 제공 키 입력 | 학교 제공 |
| OpenAI | platform.openai.com/api-keys | 신규 $5 크레딧 |
| OpenDART | opendart.fss.or.kr | 무료 |
| 네이버 OpenAPI | developers.naver.com | 무료 |
| Telegram Bot | @BotFather (앱 내) | 무료 |

### 3. Stage 1 실행 (필수 선행)

```bash
cd src/korean

python data/test_connection.py          # 연결 확인
python data/fetch_reports.py            # OpenDART 보고서 (~20분)
python data/fetch_news.py               # 네이버 뉴스 (~5분)
python data/fetch_prices.py             # 월별 종가 캐시 구축 (백테스트 사전 필요)
python data/build_testdata.py           # 테스트 데이터 구축
python data/verify_data.py              # 품질 확인

python stage1/llm_inference.py --models claude openai
python stage1/postprocess.py
python stage1/backtest.py --weight 동일가중 --long_short 롱숏
```

### 4. RAG 인덱스 구축

```bash
python rag/indexer.py    # Chroma 구축 (~3분, ~$0.01)
```

### 5. 에이전트 실행

```bash
# Streamlit 데모 (src/korean/ 디렉토리 기준)
cd src/korean
streamlit run app.py

# 또는 프로젝트 루트에서
streamlit run src/korean/app.py

# CLI 텍스트 출력
cd src/korean
python agent/graph.py --ticker 005930

# n8n용 JSON 출력
python agent/graph.py --ticker 005930 --output json
```

### 6. n8n 워크플로우 구성

n8n UI(`http://localhost:5678`)에서 위 "n8n 자동화 설계" 섹션 참조하여 워크플로우 구성

---

## 구현 로드맵

```
Week 1 — Stage 1 데이터 수집
  데이터팀 A : fetch_reports.py 실행 + 오류 수정
  데이터팀 B : fetch_news.py + fetch_prices.py + build_testdata.py
  LLM팀      : test_connection.py + API 키 확인
  백엔드팀   : pip install + Git 정리

Week 2 — LLM 추론 + RAG 인덱스 (병렬)
  LLM팀 A    : llm_inference.py 전체 실행 (~10분)
  LLM팀 B    : postprocess.py → 정확도 표
  백엔드팀 A : rag/indexer.py → Chroma 구축
  백엔드팀 B : n8n 워크플로우 1번 + Telegram 테스트

Week 3 — 에이전트 + 백테스트 (병렬)
  데이터팀 A : agent/technical.py 동작 확인
  데이터팀 B : agent/fundamental.py RAG 검색 검증
  LLM팀 A    : agent/sentiment.py Stage1 연동 확인
  LLM팀 B    : agent/synthesizer.py 프롬프트 튜닝
  백엔드팀 A : backtest.py → 차트 생성
  백엔드팀 B : app.py Streamlit + n8n 워크플로우 2번

Week 4 — 통합 + 발표
  전원 : Streamlit + n8n Telegram 데모 리허설
  PM   : 발표 자료 완성
```

---

## 팀 구성

| 팀원 | Week 1 | Week 2 | Week 3 | Week 4 |
|---|---|---|---|---|
| 데이터팀 A | fetch_reports.py | llm_inference 보조 | agent/technical.py | 데모 지원 |
| 데이터팀 B | fetch_news.py + build_testdata | postprocess.py | agent/fundamental.py | 데모 지원 |
| LLM팀 A | llm_inference.py | rag/indexer 실행 | agent/sentiment.py | 프롬프트 튜닝 |
| LLM팀 B | 환경설정 | 정확도 분석 | agent/synthesizer.py | 발표 자료 |
| 백엔드팀 A | Git 관리 | backtest.py | agent/graph.py 통합 | n8n 워크플로우 |
| 백엔드팀 B | 의존성 관리 | n8n 워크플로우 1번 | app.py + n8n 2번 | 데모 시나리오 |
| PM | 일정·문서 | 진행 점검 | 발표 자료 | 최종 발표 |

---

## 주요 결과 (실험 후 업데이트)

### LLM별 주가 방향 예측 정확도

| 모델 | 정확도 (%) | 판단 불가 비율 (%) |
|---|---|---|
| Claude Haiku 4.5 | 52.30 | 0 |
| GPT-4o-mini | 53.62 | 0 |
| 랜덤 베이스라인 | 50.0 | 0 |

### 전략별 성과 지표

| 전략 | 연환산수익률(%) | 샤프비율 | 최대낙폭(%) | KOSPI 초과수익(%) |
|---|---|---|---|---|
| Claude Haiku | 8.54 | 0.689 | -8.91 | +4.54 |
| GPT-4o-mini | 12.55 | 1.182 | -7.61 | +8.55 |
| KOSPI_proxy (벤치마크) | 3.99 | 0.262 | -11.36 | — |

> 동일가중 롱숏 전략 (벤치마크 = 30종목 동일가중 수익률, KRX 지수 API 인증 불필요)  
> 기간: 2023~2025.01 / 종목: KOSPI 30종목 / 포지션: 동일가중 롱숏

### 파이프라인 실행 현황

| 단계 | 파일 | 결과 |
|---|---|---|
| 보고서 수집 | fetch_reports.py | ✅ 304건 |
| 뉴스 수집 | fetch_news.py | ✅ 600건 |
| 종가 캐시 | fetch_prices.py | ✅ 25개월 × 30종목 |
| 테스트 데이터 | build_testdata.py | ✅ 304건 |
| LLM 추론 | llm_inference.py | ✅ Claude 52.30%, GPT 53.62% |
| 결과 파싱 | postprocess.py | ✅ parsed_predictions.xlsx |
| 백테스트 | backtest.py | ✅ outputs/korean/backtest/ |
| RAG 인덱스 | rag/indexer.py | ⏳ 미실행 (실행 시 ~3분, ~$0.01) |
| 에이전트 데모 | app.py | ⏳ 미실행 |

---

## 발표 시나리오 (15분)

```
[1부 — Stage 1 정량 결과, 5분]
  "KOSPI 30종목, 2023~2024년, 멀티 LLM 비교 실험"
  LLM별 정확도 비교표 + KOSPI 대비 누적 수익률 차트

[2부 — 라이브 데모, 7분]
  (Streamlit)
  1. "005930" 입력 → 3에이전트 + RAG 실행 (~15초)
  2. 검색된 실제 보고서·뉴스 문단 화면에 표시
  3. GPT-4o-mini 자연어 추천 출력

  (n8n + Telegram 시연)
  4. Telegram에 실시간 매수 신호 알림 도착
     "이 시스템은 매일 자동으로 분석하고 알림을 보냅니다"

[3부 — Q&A 대비, 3분]
  Q: "파인튜닝 없이 AlphaFin 구현인가?"
  A: "방법론(RAG + 멀티LLM 비교 + 롱숏 백테스트) 동일.
      한국 시장 적용 가능성 검증."

  Q: "RAG가 실제로 동작하는가?"
  A: "Chroma 검색 결과 청크를 화면에 직접 표시."

  Q: "n8n은 어떻게 연결되는가?"
  A: "--output json 플래그로 n8n이 stdout JSON 파싱 후 Telegram 발송."
```

---

## 의존성

```bash
pip install -r requirements_korean.txt
```

| 패키지 | 용도 | 비용 |
|---|---|---|
| `pykrx` | KRX 주가 (OHLCV만 사용, 시총/지수 API는 KRX 인증 필요로 미사용) | 무료 |
| `anthropic` | Claude Haiku — Stage 1 + 에이전트 sub-분석 | 학교 제공 |
| `openai` | GPT-4o-mini 합성기 + text-embedding-3-small | ~$0.05 총액 |
| `langchain` + `langchain-core` + `langchain-text-splitters` | RAG 문서 처리 | 무료 |
| `langchain-openai` + `langchain-chroma` | RAG 임베딩·검색 | 무료 |
| `chromadb` | 로컬 벡터DB | 무료 |
| `streamlit` | 대시보드 | 무료 |
| `pandas`, `numpy`, `matplotlib` | 데이터 처리·시각화 | 무료 |

> **n8n**: pip 패키지 없음. 이미 설치된 n8n UI에서 워크플로우만 구성.

---

## 선택적 확장

| 확장 | 조건 |
|---|---|
| **Supabase pgvector** | 팀 원격 공유가 필요할 때 |
| **Discord 알림** | Telegram 외 채널 원할 때 |

---

## 참고 논문

```bibtex
@inproceedings{li2024alphafin,
  title     = {AlphaFin: Benchmarking Financial Analysis with
               Retrieval-Augmented Stock-Chain Framework},
  author    = {Xiang Li and Zhenyu Li and Chen Shi and Yong Xu and
               Qing Du and Mingkui Tan and Jun Huang and Wei Lin},
  booktitle = {Proceedings of LREC-COLING 2024},
  pages     = {773--783},
  year      = {2024}
}
```

- AlphaFin GitHub: [AlphaFin-proj/AlphaFin](https://github.com/AlphaFin-proj/AlphaFin)
- OpenDART API: [opendart.fss.or.kr](https://opendart.fss.or.kr)
- pykrx: [github.com/sharebook-kr/pykrx](https://github.com/sharebook-kr/pykrx)
- Chroma: [docs.trychroma.com](https://docs.trychroma.com)
- n8n: [n8n.io](https://n8n.io)
