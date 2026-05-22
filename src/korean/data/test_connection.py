"""
데이터 수집 전 연결 상태 확인 스크립트
본격 수집 전에 반드시 실행하세요
"""
import sys
import os

_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_KOREAN_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _KOREAN_DIR)
sys.path.insert(0, _THIS_DIR)

from config import DART_API_KEY, OPENAI_API_KEY

def test_dart():
    print("=" * 50)
    print("[1] OpenDART API 연결 테스트")
    print("=" * 50)

    if not DART_API_KEY:
        print("❌ DART_API_KEY 없음 → .env 파일 확인")
        return False

    import requests
    # 삼성전자 corp_code 조회
    url = "https://opendart.fss.or.kr/api/company.json"
    resp = requests.get(url, params={"crtfc_key": DART_API_KEY, "stock_code": "005930"}, timeout=10)
    data = resp.json()

    if data.get("status") == "000":
        print(f"✅ DART 연결 성공")
        print(f"   회사명: {data.get('corp_name')}")
        print(f"   corp_code: {data.get('corp_code')}")
        return True
    else:
        print(f"❌ DART 오류: {data.get('message')}")
        return False


def test_dart_financial():
    print("\n[2] 재무 데이터 조회 테스트 (삼성전자 FY2022 사업보고서)")
    print("=" * 50)

    from fetch_reports import get_corp_code, fetch_financial_summary
    corp_code = get_corp_code("005930")
    if not corp_code:
        print("❌ corp_code 조회 실패")
        return False

    # 2023년 3월 신고된 사업보고서 → FY2022 데이터 → bsns_year="2022"
    fin = fetch_financial_summary(corp_code, "2022", "11011")
    if fin:
        revenue = int(fin.get("revenue", 0))
        print(f"✅ 재무 데이터 조회 성공")
        print(f"   매출액: {revenue:,}원")
        print(f"   영업이익: {fin.get('operating_profit', 'N/A')}원")
        print(f"   당기순이익: {fin.get('net_income', 'N/A')}원")
        return True
    else:
        print("❌ 재무 데이터 없음 (API 응답 확인 필요)")
        return False


def test_pykrx():
    print("\n[3] pykrx 주가 데이터 테스트 (삼성전자 2023년)")
    print("=" * 50)

    try:
        from pykrx import stock
        df = stock.get_market_ohlcv("20230101", "20230131", "005930")
        if df.empty:
            print("❌ pykrx 데이터 없음")
            return False
        print(f"✅ pykrx 연결 성공")
        print(f"   수집 행수: {len(df)}일")
        print(f"   마지막 종가: {df['종가'].iloc[-1]:,}원")
        return True
    except Exception as e:
        print(f"❌ pykrx 오류: {e}")
        print("   → pip install pykrx 확인")
        return False


def test_label():
    print("\n[4] 라벨 생성 테스트 (삼성전자 2023-03-31)")
    print("=" * 50)

    from fetch_prices import get_label
    label = get_label("005930", "2023-03-31")
    label_str = "상승(1)" if label == 1 else ("하락(-1)" if label == -1 else "판단불가(0)")
    print(f"✅ 라벨 생성 성공: {label_str}")
    print(f"   (2023년 4월 삼성전자 수익률 기준)")
    return label != 0


def test_openai():
    print("\n[5] OpenAI GPT-4o-mini 연결 테스트")
    print("=" * 50)

    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY 없음 → .env 파일 확인")
        return False

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "안녕하세요. 한 단어로 답하세요."}],
            max_tokens=10,
        )
        answer = resp.choices[0].message.content.strip()
        print(f"✅ OpenAI 연결 성공")
        print(f"   응답: {answer}")
        return True
    except Exception as e:
        print(f"❌ OpenAI 오류: {e}")
        if "insufficient_quota" in str(e):
            print("   → 크레딧 부족. platform.openai.com에서 충전 필요")
        elif "invalid_api_key" in str(e):
            print("   → API 키 오류. .env 파일 재확인")
        return False


if __name__ == "__main__":
    results = []
    results.append(test_dart())
    results.append(test_dart_financial())
    results.append(test_pykrx())
    results.append(test_label())
    results.append(test_openai())

    print("\n" + "=" * 50)
    passed = sum(results)
    print(f"결과: {passed}/5 통과")
    if passed == 5:
        print("✅ 모든 테스트 통과 → fetch_reports.py 실행 가능")
    elif passed >= 4:
        print("⚠️  일부 미통과 — 위 오류 메시지 확인 후 진행 가능")
    else:
        print("❌ 실패 항목 수정 후 재실행하세요")
