"""
AlphaFin Korean — Streamlit 대시보드
RAG + 멀티에이전트 실시간 종목 분석
"""
import os
import sys
import time

import streamlit as st

# Streamlit Cloud secrets → os.environ 주입 (로컬 .env 대체)
_SECRET_KEYS = [
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
    "DART_API_KEY", "NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET",
]
for _k in _SECRET_KEYS:
    if _k in st.secrets and not os.environ.get(_k):
        os.environ[_k] = st.secrets[_k]

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pykrx import stock as pykrx_stock

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "agent"))

from config import TARGET_STOCKS, CHROMA_PERSIST_DIR

# 에이전트 임포트 (절대 경로 기반, Streamlit 실행 위치 무관)
from graph import run as run_agent

# ── 페이지 설정 — 반드시 첫 번째 st.* 명령 ─────────────────────────────────
st.set_page_config(
    page_title="AlphaFin Korean",
    page_icon="📈",
    layout="wide",
)


# ── RAG 인덱스 초기 구축 (Streamlit Cloud 최초 실행 대응) ──────────────────
@st.cache_resource(show_spinner="RAG 인덱스 초기 구축 중... (최초 1회, 약 3-5분 소요)")
def _ensure_rag_index() -> tuple:
    """컬렉션이 비어 있으면 자동 빌드 — Python/OS 환경에 맞는 바이너리 생성"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        col = client.get_or_create_collection("korean_finance")
        if col.count() > 0:
            return True, None
        from rag.indexer import build_index
        build_index(reset=True)
        return True, None
    except Exception as e:
        return False, str(e)


_rag_ready, _rag_error = _ensure_rag_index()
if not _rag_ready:
    st.warning(
        f"⚠️ RAG 인덱스 구축 실패 — 펀더멘털·감성 에이전트가 문서 없이 동작합니다.\n\n"
        f"원인: `{_rag_error}`\n\n"
        "OPENAI_API_KEY가 Streamlit Secrets에 설정되어 있는지 확인하세요."
    )

# ── 한글 폰트 ──────────────────────────────────────────────────────────────
import platform
if platform.system() == "Windows":
    matplotlib.rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":
    matplotlib.rc("font", family="AppleGothic")
else:
    matplotlib.rc("font", family="NanumGothic")  # Streamlit Cloud Linux (packages.txt: fonts-nanum)
matplotlib.rc("axes", unicode_minus=False)


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_price_history(ticker: str, days: int = 90) -> pd.DataFrame:
    """최근 N일 주가 데이터"""
    end   = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y%m%d")
    df = pykrx_stock.get_market_ohlcv(start, end, ticker)
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    return df


def signal_badge(signal: int) -> str:
    if signal == 1:
        return "🟢 매수"
    elif signal == -1:
        return "🔴 매도"
    return "🟡 중립"


def signal_color(signal: int) -> str:
    return "#2ecc71" if signal == 1 else ("#e74c3c" if signal == -1 else "#f39c12")


def render_price_chart(ticker: str, stock_name: str) -> None:
    """60일 종가 + 20일 이동평균 차트"""
    df = get_price_history(ticker, 90)
    if df.empty:
        st.warning("주가 데이터를 불러올 수 없습니다.")
        return

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, df["종가"], color="#3498db", linewidth=1.5, label="종가")
    ma20 = df["종가"].rolling(20).mean()
    ax.plot(df.index, ma20, color="#e67e22", linewidth=1.2, linestyle="--", label="20일 이동평균")
    ax.fill_between(df.index, df["종가"].min(), df["종가"], alpha=0.08, color="#3498db")
    ax.set_ylabel("가격 (원)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_agent_card(title: str, icon: str, result: dict | None) -> None:
    """에이전트 결과 카드"""
    if result is None:
        st.info(f"{icon} {title}: 분석 중...")
        return

    signal  = result.get("signal", 0)
    summary = result.get("summary", "")
    conf    = result.get("confidence", None)

    color = signal_color(signal)
    badge = signal_badge(signal)

    conf_str = f" &nbsp;·&nbsp; 신뢰도 {conf:.0%}" if conf is not None else ""

    st.markdown(
        f"""
        <div style="
            border-left: 4px solid {color};
            padding: 10px 14px;
            border-radius: 4px;
            background: #1e2130;
            margin-bottom: 8px;
        ">
            <div style="font-size:13px; color:#aaa; margin-bottom:4px;">
                {icon} <b>{title}</b>
            </div>
            <div style="font-size:16px; font-weight:bold; color:{color};">
                {badge}{conf_str}
            </div>
            <div style="font-size:12px; color:#ccc; margin-top:6px; line-height:1.5;">
                {summary}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 사이드바 ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 AlphaFin Korean")
    st.caption("RAG + 멀티에이전트 한국 주식 분석")
    st.divider()

    # 종목 선택
    ticker_options = {f"{name} ({code})": code for code, name in TARGET_STOCKS.items()}
    selected_label = st.selectbox(
        "종목 선택",
        options=list(ticker_options.keys()),
        index=0,
    )
    selected_ticker = ticker_options[selected_label]
    selected_name   = TARGET_STOCKS.get(selected_ticker, selected_ticker)

    st.divider()
    analyze_btn = st.button("🔍 분석 시작", use_container_width=True, type="primary")

    st.divider()
    st.markdown(
        """
        **데이터 소스**
        - OpenDART 공시 보고서
        - 네이버 뉴스
        - KRX 주가 (pykrx)

        **에이전트**
        - 기술 에이전트 (MACD/RSI)
        - 펀더멘털 에이전트 (RAG)
        - 감성 에이전트 (RAG + Stage1)

        **합성기**
        - GPT-4o-mini
        """,
        unsafe_allow_html=False,
    )
    st.caption("⚠️ 학술 목적 전용. 실제 투자 조언 아님.")


# ── 메인 화면 ──────────────────────────────────────────────────────────────

st.title(f"📊 {selected_name} ({selected_ticker})")

# 주가 차트 (항상 표시)
with st.spinner("주가 데이터 로딩 중..."):
    render_price_chart(selected_ticker, selected_name)

st.divider()

# 분석 미실행 상태
if not analyze_btn and "last_result" not in st.session_state:
    st.info("← 왼쪽에서 종목을 선택하고 **분석 시작** 버튼을 눌러주세요.")
    st.stop()

# ── 에이전트 실행 ─────────────────────────────────────────────────────────

if analyze_btn:
    progress = st.progress(0, text="기술 분석 중...")

    with st.spinner("에이전트 분석 실행 중..."):
        try:
            start_time = time.time()
            progress.progress(10, text="🔧 에이전트 실행 중...")
            result = run_agent(selected_ticker)
            progress.progress(100, text="✅ 분석 완료")

            elapsed = time.time() - start_time
            st.session_state["last_result"] = result
            st.session_state["last_ticker"]  = selected_ticker
            st.success(f"분석 완료 ({elapsed:.1f}초)")

        except Exception as e:
            progress.empty()
            st.error(f"에이전트 실행 오류: {e}")
            st.exception(e)
            st.stop()

# ── 결과 표시 ─────────────────────────────────────────────────────────────

result = st.session_state.get("last_result")
if result is None:
    st.stop()

# 종목이 바뀐 경우 경고
if st.session_state.get("last_ticker") != selected_ticker:
    st.warning("⚠️ 종목이 변경되었습니다. 다시 분석을 실행해주세요.")

# 최종 신호 배너
final_signal = result.final_signal
signal_text  = {1: "매수", -1: "매도", 0: "중립"}.get(final_signal, "중립")
banner_color = signal_color(final_signal)
emoji        = "🟢" if final_signal == 1 else ("🔴" if final_signal == -1 else "🟡")

st.markdown(
    f"""
    <div style="
        background: {banner_color}22;
        border: 2px solid {banner_color};
        border-radius: 8px;
        padding: 16px 24px;
        text-align: center;
        margin-bottom: 16px;
    ">
        <span style="font-size:28px;">{emoji}</span>
        <span style="font-size:24px; font-weight:bold; color:{banner_color}; margin-left:12px;">
            종합 의견: {signal_text}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# 3개 에이전트 카드
col1, col2, col3 = st.columns(3)

with col1:
    tech = result.tech_result or {}
    render_agent_card("기술 에이전트", "📉", tech if tech else None)
    if tech.get("detail"):
        with st.expander("상세 지표"):
            d = tech["detail"]
            st.markdown(f"- **현재가**: {d.get('current_price', 0):,}원")
            st.markdown(f"- **MACD**: {d.get('macd_cross', '-')}")
            st.markdown(f"- **RSI**: {d.get('rsi', '-')} ({d.get('rsi_status', '-')})")
            st.markdown(f"- **20일선**: {'위' if d.get('above_ma20') else '아래'}")
            if d.get("ma60") is not None:
                st.markdown(f"- **60일선**: {'위' if d.get('above_ma60') else '아래'}")

with col2:
    fund = result.fund_result or {}
    render_agent_card("펀더멘털 에이전트", "📋", fund if fund else None)
    if fund.get("raw_output"):
        with st.expander("LLM 분석 원문"):
            st.text(fund["raw_output"])

with col3:
    sent = result.sent_result or {}
    render_agent_card("감성 에이전트", "📰", sent if sent else None)
    stage1 = (sent or {}).get("stage1_data", {})
    if stage1:
        with st.expander("Stage 1 예측 참고"):
            st.markdown(f"- **LLM 합의**: {stage1.get('agreement', '-')}")
            st.markdown(f"- **기준 보고서**: {stage1.get('report_date', '-')}")
            sigs = stage1.get("signals", {})
            for m, v in sigs.items():
                st.markdown(f"  - {m}: {'↑상승' if v==1 else ('↓하락' if v==-1 else '?')}")

st.divider()

# 자연어 투자 의견
st.subheader("💬 자연어 투자 의견 (GPT-4o-mini)")
if result.recommendation:
    st.markdown(
        f"""
        <div style="
            background: #1e2130;
            border-radius: 8px;
            padding: 16px 20px;
            line-height: 1.8;
            font-size: 14px;
            white-space: pre-wrap;
        ">{result.recommendation}</div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("합성기 출력 없음")

st.divider()

# RAG 참조 문서
if result.rag_context:
    st.subheader("📚 RAG 참조 문서 (근거 출처)")
    st.caption(f"검색된 문서 청크 {len(result.rag_context)}개")
    _source_label = {"opendart": "📋 공시보고서", "naver_news": "📰 뉴스"}
    for i, chunk in enumerate(result.rag_context[:6], 1):
        if isinstance(chunk, dict):
            _src = _source_label.get(chunk.get("source", ""), "📄 문서")
            _txt = chunk.get("text", "")
        else:
            _src = "📄 문서"
            _txt = str(chunk)
        with st.expander(f"{_src} {i} — {_txt[:40].strip()}..."):
            st.text(_txt)
else:
    st.info("RAG 인덱스가 구축되지 않았습니다. `python rag/indexer.py` 먼저 실행하세요.")

st.divider()
st.caption(
    "AlphaFin Korean | 숭실대학교 정보과학대학원 AI학과 금융AI 팀프로젝트 | "
    "학술 목적 전용 · 실제 투자 조언 아님"
)
