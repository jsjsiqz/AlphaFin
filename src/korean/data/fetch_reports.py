"""
OpenDART API를 이용한 한국 기업 공시 보고서 수집
AlphaFin의 중국 재무 리포트 수집을 대체
"""
import os
import sys
import io
import json
import time
import zipfile
import xml.etree.ElementTree as ET
import requests
import pandas as pd
from typing import Optional
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DART_API_KEY, TARGET_STOCKS, OUTPUT_DIR

DART_BASE_URL = "https://opendart.fss.or.kr/api"

# 수집 대상 보고서 종류
REPORT_TYPES = {
    "A001": "사업보고서",   # 연간
    "A002": "반기보고서",   # 반기
    "A003": "분기보고서",   # 분기 (Q1, Q3)
}

# 기업코드 로컬 캐시 경로
_CORP_CODE_CACHE = os.path.join(OUTPUT_DIR, "corp_codes.json")
_corp_code_map: Optional[dict] = None  # stock_code → corp_code 메모리 캐시


# ── 재시도 래퍼 ────────────────────────────────────────────────────────────

def _retry_request(
    url: str,
    params: dict = None,
    headers: dict = None,
    timeout: int = 10,
    max_retries: int = 3,
    delay: float = 2.0,
) -> requests.Response:
    """
    HTTP GET 재시도 래퍼 — DART API 일시 오류 대비.
    모든 재시도 실패 시 마지막 예외를 re-raise.
    """
    last_exc = None
    for i in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            if i < max_retries - 1:
                wait = delay * (i + 1)
                print(f"  [재시도 {i+1}/{max_retries}] {url.split('/')[-1]}: {e} — {wait:.0f}초 대기")
                time.sleep(wait)
    raise last_exc


# ── 기업코드 매핑 ──────────────────────────────────────────────────────────

def _load_corp_code_map() -> dict:
    """
    DART corpCode.xml 다운로드 후 stock_code → corp_code 매핑 반환.
    로컬 캐시가 있으면 재사용.
    """
    global _corp_code_map
    if _corp_code_map is not None:
        return _corp_code_map

    # 로컬 캐시 확인
    if os.path.exists(_CORP_CODE_CACHE):
        with open(_CORP_CODE_CACHE, encoding="utf-8") as f:
            _corp_code_map = json.load(f)
        return _corp_code_map

    # DART에서 전체 기업코드 ZIP 다운로드
    print("[INFO] DART 기업코드 목록 다운로드 중...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    url  = f"{DART_BASE_URL}/corpCode.xml"
    resp = _retry_request(url, params={"crtfc_key": DART_API_KEY}, timeout=60)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        xml_name = [n for n in z.namelist() if n.lower().endswith(".xml")][0]
        with z.open(xml_name) as f:
            tree = ET.parse(f)

    mapping = {}
    for item in tree.getroot().findall("list"):
        stock = (item.findtext("stock_code") or "").strip()
        corp  = (item.findtext("corp_code")  or "").strip()
        if stock:
            mapping[stock] = corp

    with open(_CORP_CODE_CACHE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)

    print(f"[INFO] 기업코드 {len(mapping)}건 캐시 저장: {_CORP_CODE_CACHE}")
    _corp_code_map = mapping
    return _corp_code_map


def get_corp_code(ticker: str) -> Optional[str]:
    """티커(stock_code)로 DART 고유번호(corp_code) 조회"""
    mapping = _load_corp_code_map()
    return mapping.get(ticker)


# ── 보고서 목록 / 재무 수집 ───────────────────────────────────────────────

def _report_nm_to_type(report_nm: str) -> Optional[str]:
    """report_nm 문자열로 보고서 유형 코드 반환"""
    if "사업보고서" in report_nm:
        return "A001"
    if "반기보고서" in report_nm:
        return "A002"
    if "분기보고서" in report_nm:
        return "A003"
    return None


def fetch_report_list(corp_code: str, start_date: str, end_date: str) -> list:
    """기업의 정기공시(사업/반기/분기보고서) 목록 조회"""
    url = f"{DART_BASE_URL}/list.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bgn_de":    start_date,
        "end_de":    end_date,
        "pblntf_ty": "A",   # 정기공시
        "page_count": 40,
    }
    try:
        resp = _retry_request(url, params=params, timeout=10)
        data = resp.json()
    except Exception as e:
        print(f"  [WARN] 보고서 목록 조회 실패 ({corp_code}): {e}")
        return []

    if data.get("status") != "000":
        return []

    results = []
    for r in data.get("list", []):
        rpt_nm   = r.get("report_nm", "")
        rpt_type = _report_nm_to_type(rpt_nm)
        if rpt_type:
            r["pblntf_detail_ty"] = rpt_type
            results.append(r)
    return results


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
        "crtfc_key":  DART_API_KEY,
        "corp_code":  corp_code,
        "bsns_year":  year,
        "reprt_code": report_code,
        "fs_div":     "CFS",   # 연결재무제표
    }
    try:
        resp = _retry_request(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") != "000":
            return {}

        result = {}
        for item in data.get("list", []):
            account = item.get("account_nm", "")
            raw     = item.get("thstrm_amount", "") or ""
            amount  = raw.replace(",", "").strip()
            if not amount or amount == "-":
                continue  # 빈 값·대시는 저장하지 않음 (N/A로 표시됨)
            if "매출액" in account and "revenue" not in result:
                result["revenue"] = amount
            elif "영업이익" in account and "operating_profit" not in result:
                result["operating_profit"] = amount
            elif "당기순이익" in account and "net_income" not in result:
                result["net_income"] = amount
        return result
    except Exception:
        return {}


# ── 전체 수집 ─────────────────────────────────────────────────────────────

def build_report_db(start_date: str = "20230101", end_date: str = None) -> list[dict]:
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
    if end_date is None:
        from datetime import date as _date
        end_date = _date.today().strftime("%Y%m%d")

    save_dir = os.path.join(OUTPUT_DIR, "reports")
    os.makedirs(save_dir, exist_ok=True)

    all_reports = []

    for ticker, name in tqdm(TARGET_STOCKS.items(), desc="보고서 수집"):
        corp_code = get_corp_code(ticker)
        if not corp_code:
            print(f"[WARN] corp_code 조회 실패: {name}({ticker})")
            continue

        reports = fetch_report_list(
            corp_code,
            start_date.replace("-", ""),
            end_date.replace("-", ""),
        )
        time.sleep(0.3)  # API 속도 제한 준수

        for r in reports:
            rpt_type = r.get("pblntf_detail_ty", "")
            if rpt_type not in REPORT_TYPES:
                continue

            rcept_dt = r.get("rcept_dt", "")
            if not rcept_dt:
                continue

            report_date  = f"{rcept_dt[:4]}-{rcept_dt[4:6]}-{rcept_dt[6:8]}"
            rcept_year   = int(rcept_dt[:4])
            rcept_month  = int(rcept_dt[4:6])

            # 사업보고서(A001)는 해당 연도에 전년도 실적을 신고
            fin_year   = str(rcept_year - 1) if rpt_type == "A001" else str(rcept_year)
            reprt_code = get_reprt_code(rpt_type, rcept_month)

            fin_summary = fetch_financial_summary(corp_code, fin_year, reprt_code)
            time.sleep(0.2)

            all_reports.append({
                "ticker":            ticker,
                "stock_name":        name,
                "report_date":       report_date,
                "report_type":       REPORT_TYPES[rpt_type],
                "rcept_no":          r.get("rcept_no", ""),
                "financial_summary": fin_summary,
            })

    save_path = os.path.join(save_dir, "reports_raw.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] {len(all_reports)}건 보고서 저장: {save_path}")
    return all_reports


if __name__ == "__main__":
    build_report_db()
