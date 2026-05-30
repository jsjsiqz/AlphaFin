"""
LangChain Document Loader
OpenDART 보고서 + 네이버 뉴스 → LangChain Document 변환
"""
import os
import sys
import json
from email.utils import parsedate_to_datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_KOREAN_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _KOREAN_DIR)

from config import OUTPUT_DIR


def _fmt_krw(val) -> str:
    """원화 금액을 억원/조원 단위로 포맷 — LLM이 읽기 쉬운 형태"""
    if val is None or val == "" or val == "N/A":
        return "N/A"
    try:
        n = int(str(val).replace(",", "").split(".")[0])
        eok = n // 100_000_000
        if abs(eok) >= 10_000:
            jo = eok // 10_000
            rem = eok % 10_000
            return f"{jo:,}조 {rem:,}억원"
        return f"{eok:,}억원"
    except (ValueError, TypeError):
        return "N/A"


def _format_naver_date(date_str: str) -> str:
    """네이버 API RFC 2822 날짜를 YYYY-MM-DD로 변환. 실패 시 앞 10자 반환."""
    if not date_str:
        return ""
    try:
        return parsedate_to_datetime(date_str).strftime("%Y-%m-%d")
    except Exception:
        return date_str[:10]


def load_reports() -> list:
    """OpenDART 보고서 → LangChain Document 리스트"""
    path = os.path.join(OUTPUT_DIR, "reports", "reports_raw.json")
    if not os.path.exists(path):
        print(f"[WARN] {path} 없음 → fetch_reports.py 먼저 실행")
        return []

    try:
        with open(path, encoding="utf-8") as f:
            reports = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[ERROR] reports_raw.json 읽기 실패: {e}")
        return []

    documents = []
    for r in reports:
        ticker = str(r.get("ticker", "")).strip()
        if not ticker:
            continue  # ticker 없는 문서는 RAG 필터를 오염시킴

        fin = r.get("financial_summary", {})
        content = (
            f"{r.get('stock_name','')}({ticker}) "
            f"{r.get('report_date','')} {r.get('report_type','')}\n"
            f"매출액: {_fmt_krw(fin.get('revenue'))}\n"
            f"영업이익: {_fmt_krw(fin.get('operating_profit'))}\n"
            f"당기순이익: {_fmt_krw(fin.get('net_income'))}"
        )
        documents.append(Document(
            page_content=content,
            metadata={
                "ticker":      ticker,
                "stock_name":  r.get("stock_name", ""),
                "report_date": r.get("report_date", ""),
                "report_type": r.get("report_type", ""),
                "source":      "opendart",
            },
        ))

    print(f"[INFO] 보고서 {len(documents)}건 로드")
    return documents


def load_news() -> list:
    """네이버 뉴스 → LangChain Document 리스트"""
    path = os.path.join(OUTPUT_DIR, "news", "news_raw.json")
    if not os.path.exists(path):
        print("[WARN] news_raw.json 없음 → fetch_news.py 먼저 실행 (선택사항)")
        return []

    try:
        with open(path, encoding="utf-8") as f:
            news = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[ERROR] news_raw.json 읽기 실패: {e}")
        return []

    documents = []
    skipped = 0
    for n in news:
        ticker = str(n.get("ticker", "")).strip()
        if not ticker:
            skipped += 1
            continue  # ticker 없으면 RAG 필터가 이 문서를 잘못 반환할 수 있음

        pub_date = _format_naver_date(n.get("pub_date", ""))
        date_prefix = f"[{pub_date}] " if pub_date else ""
        content = f"{date_prefix}{n.get('title', '')}\n{n.get('description', '')}"
        if not content.strip():
            skipped += 1
            continue

        documents.append(Document(
            page_content=content,
            metadata={
                "ticker":     ticker,
                "stock_name": n.get("stock_name", ""),
                "pub_date":   n.get("pub_date", ""),
                "source":     "naver_news",
            },
        ))

    if skipped:
        print(f"[WARN] 뉴스 {skipped}건 스킵 (ticker 없음 또는 빈 내용)")
    print(f"[INFO] 뉴스 {len(documents)}건 로드")
    return documents


def split_documents(documents: list) -> list:
    """
    RAG 청킹: RecursiveCharacterTextSplitter
    chunk_size=500 — LLM 컨텍스트 효율과 검색 정밀도의 균형
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] 청크 {len(chunks)}개 생성")
    return chunks
