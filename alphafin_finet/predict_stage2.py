"""
[샘플] Stage 2 (CoT) 결과물 로드 → 삼성전자 주가 방향 예측 예제

train_stage2_cot.py 로 학습된 결과물을 읽어, 학습 때와 동일한 프롬프트 포맷
(### 지시 / ### 입력 / ### 응답)으로 추론한다.

기본은 Stage 2의 merged_16bit(병합 모델)을 로드한다. 없으면 final/(LoRA 어댑터)을
Stage 1 merged_16bit 위에 얹어 로드하는 방식으로 폴백한다.

실행:
    python predict_stage2.py

생성 옵션은 아래 "생성 설정" 상수에서 직접 수정한다.
"""
import os

import torch
from unsloth import FastLanguageModel


HERE        = os.path.dirname(os.path.abspath(__file__))
STAGE2_DIR  = os.path.join(HERE, "outputs", "bllossom_stage2_cot")
MERGED_DIR  = os.path.join(STAGE2_DIR, "merged_16bit")   # 우선 로드 (병합 모델)
FINAL_DIR   = os.path.join(STAGE2_DIR, "final")          # 폴백 (LoRA 어댑터)
STAGE1_MERGED = os.path.join(HERE, "outputs", "bllossom_stage1", "merged_16bit")

MAX_SEQ_LENGTH = 3072   # 학습(train_stage2_cot.py)과 동일

# ── 생성 설정 (실행 옵션을 코드 안에서 직접 조정) ────────────────────
MAX_NEW_TOKENS = 500     # 생성 최대 토큰 수
TEMPERATURE    = 0.3     # 0 이면 greedy
TOP_P          = 0.9

# 학습 때와 동일한 instruction (단계별 CoT 추론 요구)
INSTRUCTION = (
    "다음 증권사 리포트와 시장 데이터를 분석하여 해당 종목의 향후 1개월 주가 방향을 "
    "예측하시오. 펀더멘털 분석 → 기술적 분석 → 종합 판단 순서로 단계별 근거를 서술한 뒤, "
    "마지막에 '상승' 또는 '하락'과 확신도(0~1)를 제시하시오."
)

# 삼성전자(005930) 예측용 입력 — 실제로는 리포트/시장데이터를 채워 넣는다.
INPUT = """[리포트 메타]
발행일: 2026-06-02
증권사: 예시증권
애널리스트: 홍길동
종목명: 삼성전자 (005930)
투자의견: BUY
목표주가: 95,000원
발행시 주가: 78,000원

[리포트 본문]
HBM3E 12단 양산이 본격화되며 메모리 업황 회복 국면 진입. 2026년 DS부문 영업이익
개선 폭이 시장 기대치를 상회할 것으로 전망. 파운드리는 적자 축소 흐름.

[시장 데이터]
직전 20거래일 종가(원): 74100, 74500, 75200, 74800, 75600, 76300, 76000, 77100,
77800, 77500, 78200, 79000, 78600, 78400, 79100, 78900, 78300, 78000, 78500, 78000
"""


def build_prompt(instruction: str, user_input: str) -> str:
    """학습(format_prompt)과 동일한 포맷. 응답은 모델이 채운다."""
    return (
        f"### 지시:\n{instruction}\n\n"
        f"### 입력:\n{user_input}\n\n"
        f"### 응답:\n"
    )


def load_model():
    """Stage 2 merged_16bit 우선, 없으면 Stage1 merged + Stage2 LoRA 폴백."""
    if os.path.isdir(MERGED_DIR):
        print(f"[load] Stage 2 병합 모델 로드: {MERGED_DIR}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = MERGED_DIR,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype          = None,
            load_in_4bit   = True,
        )
    elif os.path.isdir(FINAL_DIR) and os.path.isdir(STAGE1_MERGED):
        print(f"[load] Stage 1 병합 모델 + Stage 2 LoRA 어댑터 로드")
        print(f"       base   : {STAGE1_MERGED}")
        print(f"       adapter: {FINAL_DIR}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = STAGE1_MERGED,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype          = None,
            load_in_4bit   = True,
        )
        model.load_adapter(FINAL_DIR)
    else:
        raise SystemExit(
            f"\n✗ Stage 2 결과물을 찾을 수 없습니다.\n"
            f"  먼저 `python train_stage2_cot.py` 를 실행하세요.\n"
            f"  (기대 경로: {MERGED_DIR} 또는 {FINAL_DIR})\n"
        )

    FastLanguageModel.for_inference(model)   # Unsloth 추론 모드 (2x 가속)
    return model, tokenizer


def main():
    model, tokenizer = load_model()

    prompt = build_prompt(INSTRUCTION, INPUT)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("\n" + "=" * 60)
    print("프롬프트: 삼성전자(005930) 향후 1개월 주가 방향 예측")
    print("=" * 60)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            temperature    = TEMPERATURE,
            top_p          = TOP_P,
            do_sample      = TEMPERATURE > 0,
            repetition_penalty = 1.1,
            eos_token_id   = tokenizer.eos_token_id,
            pad_token_id   = tokenizer.eos_token_id,
        )

    # 프롬프트 이후 생성된 응답만 디코드
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True)

    print("\n[모델 응답]\n")
    print(answer.strip())
    print()


if __name__ == "__main__":
    main()
