"""
벡터 스토어 — Chroma (로컬 무료)
LangChain 통합 어댑터

지연 초기화: API 키 없이도 import 성공.
실제 임베딩/검색 시에만 OpenAI API 호출.
"""
import os
import sys

_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_KOREAN_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _KOREAN_DIR)

from config import CHROMA_PERSIST_DIR

_embedding = None


def _get_embedding():
    """OpenAI Embedding 인스턴스 (지연 초기화 — 호출 시점에 키 읽기)"""
    global _embedding
    if _embedding is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY가 설정되지 않았습니다.\n"
                ".env 파일 또는 Streamlit Secrets에 OPENAI_API_KEY를 추가하세요."
            )
        from langchain_openai import OpenAIEmbeddings
        _embedding = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key,
        )
    return _embedding


def get_vectorstore():
    """Chroma 벡터스토어 반환 (로컬 퍼시스턴트)"""
    from langchain_chroma import Chroma
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return Chroma(
        collection_name="korean_finance",
        embedding_function=_get_embedding(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def get_retriever(ticker: str = None, source: str = None, k: int = 3):
    """
    메타데이터 필터를 포함한 LangChain Retriever 반환

    Args:
        ticker: 종목 코드 필터 (예: "005930")
        source: 문서 유형 필터 ("opendart" | "naver_news")
        k:      반환 문서 수
    """
    vs = get_vectorstore()
    search_kwargs: dict = {"k": k}

    conditions = []
    if ticker:
        conditions.append({"ticker": {"$eq": ticker}})
    if source:
        conditions.append({"source": {"$eq": source}})

    if len(conditions) == 1:
        search_kwargs["filter"] = conditions[0]
    elif len(conditions) > 1:
        search_kwargs["filter"] = {"$and": conditions}

    return vs.as_retriever(search_kwargs=search_kwargs)
