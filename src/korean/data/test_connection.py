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
    print("[1] OpenDART API 연결 테스트 (기업코드 목록 다운로드)")
    print("=" * 50)

    if not DART_API_KEY:
        print("❌ DART_API_KEY 없음 → .env 파일 확인")
        return False

    from fetch_reports import get_corp_code, _load_corp_code_map
    try:
        mapping = _load_corp_code_map()
        corp_code = mapping.get("005930")
        if corp_code:
            print(f"✅ DART 연결 성공")
            print(f"   삼성전자 corp_code: {corp_code}")
            print(f"   전체 기업코드 수: {len(mapping)}건")
            return True
        else:
            print("❌ 삼성전자 corp_code 조회 실패")
            return False
    except Exception as e:
        print(f"❌ DART 오류: {e}")
        return False


def test_dart_financial():
    print("\n[2] 재무 데이터 조회 테스트 (삼성전자 FY2022 사업보고서)")
    print("=" * 50)

    from fetch_reports import get_corp_code, fetch_financial_summary
    corp_code = get_corp_code("005930")
    if not corp_code:
        print("❌ corp_code 조회 실패 (test_dart 먼저 통과 필요)")
        return False

    fin = fetch_financial_summary(corp_code, "2022", "11011")
    if fin:
        revenue_str = fin.get("revenue", "0").replace(",", "")
        revenue = int(revenue_str) if revenue_str.lstrip("-").isdigit() else 0
        print(f"✅ 재무 데이터 조회 성공")
        print(f"   매출액: {revenue:,}원")
        print(f"   영업이익: {fin.get('operating_profit', 'N/A')}원")
        print(f"   당기순이익: {fin.get('net_income', 'N/A')}원")
        return True
    else:
        print("❌ 재무 데이터 없음")
        return False


def test_pykrx():
    print("\n[3] pykrx 주가 데이터 테스트 (삼성전자 2023년)")
    print("=" * 50)
    print("  ※ 인증 안내:")
    print("    - get_market_ohlcv : 인증 불필요 ✅ (주가 수집에 사용)")
    print("    - get_market_cap   : KRX_ID/KRX_PW 필요 ❌ (동일가중으로 대체)")
    print("    - get_index_ohlcv  : KRX_ID/KRX_PW 필요 ❌ (30종목 프록시로 대체)")

    try:
        from pykrx import stock
        df = stock.get_market_ohlcv("20230101", "20230131", "005930")
        if df.empty:
            print("❌ pykrx 데이터 없음")
            return False
        print(f"✅ pykrx(OHLCV) 연결 성공")
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
