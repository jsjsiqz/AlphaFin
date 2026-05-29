"""
RAG 인덱스 구축 실행 스크립트
Stage 1 완료(fetch_reports.py + fetch_news.py) 후 실행

사용법:
    cd src/korean
    python rag/indexer.py           # 전체 인덱스 구축 (중복 자동 스킵)
    python rag/indexer.py --reset   # 기존 인덱스 삭제 후 재구축
"""
import os
import sys
import hashlib

_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_KOREAN_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _KOREAN_DIR)
sys.path.insert(0, _THIS_DIR)

from loader import load_reports, load_news, split_documents
from vectorstore import get_vectorstore


def _doc_id(doc) -> str:
    """내용 + 메타데이터 기반 결정론적 ID — 동일 문서 재인덱싱 시 upsert(중복 방지)"""
    key = (
        doc.metadata.get("ticker", "")
        + doc.metadata.get("report_date", doc.metadata.get("pub_date", ""))
        + doc.metadata.get("source", "")
        + doc.page_content[:80]
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()


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

    # ID 기준 전역 중복 제거 — 같은 배치 내 혹은 배치 간 동일 ID가 있으면 Chroma가 오류를 냄
    seen: dict[str, int] = {}
    unique_chunks = []
    for chunk in chunks:
        cid = _doc_id(chunk)
        if cid not in seen:
            seen[cid] = 1
            unique_chunks.append((cid, chunk))
    dup_count = len(chunks) - len(unique_chunks)
    if dup_count:
        print(f"      [INFO] 중복 청크 {dup_count}개 제거 (동일 문서 재인덱싱)")

    # 배치 저장 — 결정론적 ID로 upsert하여 중복 방지 (API 속도 제한 대비)
    batch_size = 100
    for i in range(0, len(unique_chunks), batch_size):
        batch_pairs = unique_chunks[i : i + batch_size]
        ids   = [p[0] for p in batch_pairs]
        batch = [p[1] for p in batch_pairs]
        vs.add_documents(batch, ids=ids)
        done = min(i + batch_size, len(unique_chunks))
        print(f"      저장: {done}/{len(unique_chunks)}건")

    print(f"\n[완료] 인덱스 구축 완료")
    print(f"       보고서 {len(report_docs)}건 + 뉴스 {len(news_docs)}건 = 총 {len(unique_chunks)}청크 (원본 {len(chunks)}개 → 중복제거 후)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="기존 인덱스 삭제 후 재구축")
    args = parser.parse_args()
    build_index(reset=args.reset)
