"""
멀티 LLM 주가 방향 예측 (AlphaFin stockgpt_inf.py의 한국판)
Claude(Haiku) / OpenAI GPT-4o-mini 멀티 LLM 주가 방향 예측
"""
import os
import sys
import json
import time
import argparse
from tqdm import tqdm

import anthropic
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    MODELS, OUTPUT_DIR
)

# 모델별 최소 대기 시간 (API RPM 제한 기준)
# Claude Haiku (학교 제공):    200 RPM → 1초 (여유 있게 설정)
# OpenAI GPT-4o-mini:         500 RPM → 1초
MODEL_DELAYS = {
    "claude": 1.0,
    "openai": 1.0,
}


# ── LLM 클라이언트 초기화 ──────────────────────────────────────────────────

def init_clients() -> dict:
    clients = {}

    if ANTHROPIC_API_KEY:
        clients["claude"] = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if OPENAI_API_KEY:
        clients["openai"] = OpenAI(api_key=OPENAI_API_KEY)

    return clients


# ── LLM 호출 함수 ──────────────────────────────────────────────────────────

def call_claude(client, prompt: str) -> str:
    try:
        resp = client.messages.create(
            model=MODELS["claude"],
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        return f"[ERROR] {e}"


def call_openai(client, prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODELS["openai"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {e}"


def call_llm(model_name: str, clients: dict, prompt: str) -> str:
    if model_name == "claude":
        return call_claude(clients["claude"], prompt)
    elif model_name == "openai":
        return call_openai(clients["openai"], prompt)
    return "[ERROR] unknown model"


# ── 추론 파이프라인 ────────────────────────────────────────────────────────

def build_prompt(instruction: str, input_text: str) -> str:
    """AlphaFin의 ChatGLMPrompt()와 동일한 역할"""
    return f"{instruction}\n\n{input_text}"


def load_done_keys(save_path: str) -> set:
    """이미 처리된 (ticker, date) 쌍 반환 — 재실행 시 중복 방지"""
    done = set()
    if not os.path.exists(save_path):
        return done
    with open(save_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["ticker"], r["date"]))
            except Exception:
                pass
    return done


def run_inference(
    data_path: str,
    model_names: list,
    output_dir: str = None,
    delay: float = 0.0,
) -> list:
    """
    testdata를 읽어 멀티 LLM으로 추론 실행
    AlphaFin의 prompt_eval()과 동일한 역할

    Args:
        data_path:   korean_testdata.json 경로
        model_names: ["claude", "openai"] 중 선택
        output_dir:  결과 저장 디렉토리
        delay:       모델별 기본 딜레이에 추가할 시간(초), 보통 0으로 유지
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, "llm_predictions.jsonl")

    with open(data_path, encoding="utf-8") as f:
        testdata = json.load(f)

    clients = init_clients()
    available = [m for m in model_names if m in clients]
    if not available:
        raise RuntimeError("사용 가능한 LLM 클라이언트가 없습니다. API 키를 확인하세요.")

    # 이미 처리된 항목 확인 (중단 후 재실행 시 스킵)
    done_keys = load_done_keys(save_path)
    if done_keys:
        print(f"[INFO] 이미 처리된 {len(done_keys)}건 스킵 (이어서 실행)")

    remaining = [s for s in testdata if (s["ticker"], s["date"]) not in done_keys]
    print(f"[INFO] 처리 대상: {len(remaining)}건 / 전체: {len(testdata)}건, 모델: {available}")

    results = []
    for sample in tqdm(remaining, desc="LLM 추론"):
        prompt = build_prompt(sample["instruction"], sample["input"])
        result = {
            "ticker":       sample["ticker"],
            "stock_name":   sample["stock_name"],
            "date":         sample["date"],
            "ground_truth": sample["output"],
            "label":        sample["label"],
        }

        for model_name in available:
            output = call_llm(model_name, clients, prompt)
            result[model_name] = output
            wait = MODEL_DELAYS.get(model_name, 1.0) + delay
            time.sleep(wait)

        results.append(result)

        with open(save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n[완료] {len(results)}건 추론 결과 저장: {save_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=f"{OUTPUT_DIR}/korean_testdata.json")
    parser.add_argument("--models", nargs="+", default=["claude", "openai"])
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    run_inference(args.data_path, args.models, args.output_dir, args.delay)


if __name__ == "__main__":
    main()
