"""
에이전트 공유 상태 정의
dataclass로 단순하게 — LangGraph 불필요
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentState:
    # 입력
    ticker:        str
    stock_name:    str   = ""
    current_price: float = 0.0

    # 각 에이전트 출력
    tech_result:  Optional[dict] = None
    fund_result:  Optional[dict] = None
    sent_result:  Optional[dict] = None

    # RAG 검색 컨텍스트 (합성기·Streamlit 표시용)
    rag_context:  list = field(default_factory=list)

    # 최종 결과
    final_signal:   int = 0    # -1(매도) / 0(중립) / 1(매수)
    recommendation: str = ""   # GPT-4o-mini 자연어 추천
