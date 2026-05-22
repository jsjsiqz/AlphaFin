"""
벡터 스토어 — Chroma (로컬 무료)
LangChain 통합 어댑터

지연 초기화: API 키 없이도 import 성공.
실제 임베딩/검색 시에만 OpenAI API 호출.

※ Supabase pgvector 선택적 확장은 하단 주석 참고
"""
import os
import sys

_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_KOREAN_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _KOREAN_DIR)

from config import OPENAI_API_KEY, CHROMA_PERSIST_DIR

# 지연 초기화 — import 시점에 API 호출 없음
_embedding  = None
_vectorstore = None


def _get_embedding():
    """OpenAI Embedding 인스턴스 (지연 초기화)"""
    global _embedding
    if _embedding is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY가 설정되지 않았습니다.\n"
                ".env 파일에 OPENAI_API_KEY를 추가하세요."
            )
        from langchain_openai import OpenAIEmbeddings
        _embedding = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY,
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

    filter_dict: dict = {}
    if ticker:
        filter_dict["ticker"] = ticker
    if source:
        filter_dict["source"] = source
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    return vs.as_retriever(search_kwargs=search_kwargs)


# ── 선택적 확장: Supabase pgvector ────────────────────────────────────────
# 팀 공유 또는 클라우드 배포가 필요할 때만 사용.
# 추가 패키지: pip install supabase langchain-community
#
# def get_supabase_vectorstore():
#     from supabase import create_client
#     from langchain_community.vectorstores import SupabaseVectorStore
#     from config import SUPABASE_URL, SUPABASE_KEY
#     client = create_client(SUPABASE_URL, SUPABASE_KEY)
#     return SupabaseVectorStore(
#         client=client,
#         embedding=_get_embedding(),
#         table_name="documents",
#         query_name="match_documents",
#     )
