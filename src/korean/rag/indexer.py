"""
RAG 인덱스 구축 실행 스크립트
Stage 1 완료(fetch_reports.py + fetch_news.py) 후 실행

사용법:
    cd src/korean
    python rag/indexer.py           # 전체 인덱스 구축
    python rag/indexer.py --reset   # 기존 인덱스 삭제 후 재구축
"""
import os
import sys

_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_KOREAN_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _KOREAN_DIR)
sys.path.insert(0, _THIS_DIR)

from loader import load_reports, load_news, split_documents
from vectorstore import get_vectorstore


def build_index(reset: bool = False) -> None:
    """전체 문서를 임베딩하여 Chroma 벡터스토어에 저장"""
    print("\n[RAG Indexer] Chroma 로컬 벡터스토어")
    print("=" * 50)

    # 1. 문서 로드
    print("\n[1/3] 문서 로드 중...")
    report_docs = load_reports()
    news_docs   = load_news()
    all_docs    = report_docs + news_docs

    if not all_docs:
        print("[ERROR] 로드된 문서 없음")
        print("        fetch_reports.py 와 fetch_news.py 먼저 실행하세요")
        return

    # 2. 청크 분할
    print("\n[2/3] 청크 분할 중...")
    chunks = split_documents(all_docs)

    # 3. 벡터 스토어 저장
    print("\n[3/3] 임베딩 + 저장 중... (OpenAI text-embedding-3-small, ~$0.01)")
    vs = get_vectorstore()

    if reset:
        vs.delete_collection()
        vs = get_vectorstore()

    # 배치 저장 (API 속도 제한 대비)
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vs.add_documents(batch)
        done = min(i + batch_size, len(chunks))
        print(f"      저장: {done}/{len(chunks)}건")

    print(f"\n[완료] 인덱스 구축 완료")
    print(f"       보고서 {len(report_docs)}건 + 뉴스 {len(news_docs)}건 = 총 {len(chunks)}청크")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="기존 인덱스 삭제 후 재구축")
    args = parser.parse_args()
    build_index(reset=args.reset)
