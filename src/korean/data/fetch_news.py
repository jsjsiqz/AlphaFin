"""
네이버 검색 OpenAPI를 이용한 금융 뉴스 수집
RAG 지식베이스의 감성 정보 소스
"""
import os
import sys
import re
import json
import time
import requests
from tqdm import tqdm

_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_KOREAN_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _KOREAN_DIR)

from config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, TARGET_STOCKS, OUTPUT_DIR

NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"


def clean_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def fetch_news(stock_name: str, display: int = 20) -> list:
    """종목명으로 최신 뉴스 수집"""
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print("[WARN] 네이버 API 키 없음 → fetch_news 스킵")
        return []

    headers = {
        "X-Naver-Client-Id":     NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {
        "query":   f"{stock_name} 주가 실적 공시",
        "display": display,
        "sort":    "date",
    }
    try:
        resp = requests.get(NAVER_NEWS_URL, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [
            {
                "title":       clean_html(item.get("title", "")),
                "description": clean_html(item.get("description", "")),
                "pub_date":    item.get("pubDate", ""),
                "link":        item.get("link", ""),
            }
            for item in items
        ]
    except Exception as e:
        print(f"[ERROR] {stock_name} 뉴스 수집 실패: {e}")
        return []


def build_news_db(save_dir: str = None) -> list:
    """전체 종목 뉴스 수집 → news_raw.json 저장"""
    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, "news")
    os.makedirs(save_dir, exist_ok=True)

    all_news = []
    for ticker, name in tqdm(TARGET_STOCKS.items(), desc="뉴스 수집"):
        articles = fetch_news(name)
        for article in articles:
            article["ticker"]     = ticker
            article["stock_name"] = name
        all_news.extend(articles)
        time.sleep(0.1)

    save_path = os.path.join(save_dir, "news_raw.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_news, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] {len(all_news)}건 뉴스 저장: {save_path}")
    return all_news


if __name__ == "__main__":
    build_news_db()
