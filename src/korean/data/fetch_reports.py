"""
OpenDART API를 이용한 한국 기업 공시 보고서 수집
AlphaFin의 중국 재무 리포트 수집을 대체
"""
import os
import sys
import json
import time
import requests
import pandas as pd
from typing import Optional
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DART_API_KEY, TARGET_STOCKS, OUTPUT_DIR

DART_BASE_URL = "https://opendart.fss.or.kr/api"

# 수집 대상 보고서 종류
REPORT_TYPES = {
    "A001": "사업보고서",       # 연간
    "A002": "반기보고서",       # 반기
    "A003": "분기보고서",       # 분기 (Q1, Q3)
}


def get_corp_code(ticker: str) -> Optional[str]:
    """티커로 DART 고유번호 조회"""
    url = f"{DART_BASE_URL}/company.json"
    params = {"crtfc_key": DART_API_KEY, "stock_code": ticker}
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    if data.get("status") == "000":
        return data.get("corp_code")
    return None


def fetch_report_list(corp_code: str, start_date: str, end_date: str) -> list:
    """기업의 정기공시(A001/A002/A003) 목록 조회"""
    url = f"{DART_BASE_URL}/list.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bgn_de": start_date,
        "end_de": end_date,
        "pblntf_ty": "A",   # 정기공시 유형 전체
        "page_count": 40,
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    if data.get("status") == "000":
        # A001(사업보고서), A002(반기), A003(분기)만 필터링
        return [r for r in data.get("list", [])
                if r.get("pblntf_detail_ty") in REPORT_TYPES]
    return []


def get_reprt_code(rpt_type: str, rcept_month: int) -> str:
    """
    DART fnlttSinglAcntAll API용 보고서 코드 결정

    A003(분기보고서)는 Q1(5월 전후 신고)과 Q3(11월 전후 신고)로 구분
    - Q1: reprt_code = 11013
    - Q3: reprt_code = 11014
    """
    if rpt_type == "A001":
        return "11011"
    if rpt_type == "A002":
        return "11012"
    if rpt_type == "A003":
        return "11013" if rcept_month <= 7 else "11014"
    return "11011"


def fetch_financial_summary(corp_code: str, year: str, report_code: str = "11011") -> dict:
    """
    주요 재무 지표 수집 (매출, 영업이익, 당기순이익)
    report_code: 11011=사업보고서, 11012=반기, 11013=Q1, 11014=Q3
    """
    url = f"{DART_BASE_URL}/fnlttSinglAcntAll.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bsns_year": year,
        "reprt_code": report_code,
        "fs_div": "CFS",  # 연결재무제표
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") != "000":
            return {}

        result = {}
        for item in data.get("list", []):
            account = item.get("account_nm", "")
            amount = item.get("thstrm_amount", "0").replace(",", "")
            if "매출액" in account:
                result["revenue"] = amount
            elif "영업이익" in account:
                result["operating_profit"] = amount
            elif "당기순이익" in account:
                result["net_income"] = amount
        return result
    except Exception:
        return {}


def build_report_db(start_date: str = "20230101", end_date: str = "20241231") -> list[dict]:
    """
    전체 종목의 공시 보고서를 수집하여 AlphaFin testdata 형식으로 반환

    반환 형식:
    {
        "ticker": "005930",
        "stock_name": "삼성전자",
        "report_date": "2023-03-31",
        "report_type": "사업보고서",
        "financial_summary": {...},
        "rcept_no": "...",
    }
    """
    save_dir = os.path.join(OUTPUT_DIR, "reports")
    os.makedirs(save_dir, exist_ok=True)

    all_reports = []

    for ticker, name in tqdm(TARGET_STOCKS.items(), desc="보고서 수집"):
        corp_code = get_corp_code(ticker)
        if not corp_code:
            print(f"[WARN] corp_code 조회 실패: {name}({ticker})")
            continue

        reports = fetch_report_list(corp_code, start_date.replace("-", ""), end_date.replace("-", ""))
        time.sleep(0.3)  # API 속도 제한 준수

        for r in reports:
            rpt_type = r.get("pblntf_detail_ty", "")
            if rpt_type not in REPORT_TYPES:
                continue

            rcept_dt = r.get("rcept_dt", "")
            if not rcept_dt:
                continue

            report_date = f"{rcept_dt[:4]}-{rcept_dt[4:6]}-{rcept_dt[6:8]}"
            rcept_year = int(rcept_dt[:4])
            rcept_month = int(rcept_dt[4:6])

            # 사업보고서(A001)는 해당 연도에 전년도 실적을 신고
            # 예: 2023년 3월 신고 → FY2022 재무 데이터
            fin_year = str(rcept_year - 1) if rpt_type == "A001" else str(rcept_year)
            reprt_code = get_reprt_code(rpt_type, rcept_month)

            fin_summary = fetch_financial_summary(corp_code, fin_year, reprt_code)
            time.sleep(0.2)

            entry = {
                "ticker": ticker,
                "stock_name": name,
                "report_date": report_date,
                "report_type": REPORT_TYPES[rpt_type],
                "rcept_no": r.get("rcept_no", ""),
                "financial_summary": fin_summary,
            }
            all_reports.append(entry)

    save_path = os.path.join(save_dir, "reports_raw.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] {len(all_reports)}건 보고서 저장: {save_path}")
    return all_reports


if __name__ == "__main__":
    build_report_db()
