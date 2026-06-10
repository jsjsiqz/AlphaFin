"""
[샘플] Llama-3-Korean-Bllossom-8B QLoRA fine-tuning — Stage 1 (단순 라벨)

AlphaFin 한국판 재현 프로젝트의 Stage 1 학습 코드 (GitHub 공개용 최소 예제).
output = "up" / "down" (1토큰) 을 학습한다.

  - 데이터:   sample/data/stage1_sample.jsonl  (최신 2건 샘플)
  - 베이스:   MLP-KTLim/llama-3-Korean-Bllossom-8B
  - 기법:     LoRA (16-bit) + Unsloth  (LOAD_IN_4BIT=True 로 4-bit QLoRA 전환 가능)
  - 타겟 GPU: RTX 3090 / 4090 / A6000 (24GB+)

⚠️ 이 코드는 파이프라인을 보여주기 위한 샘플이다. 단순 라벨(1토큰) 학습은
   소량 데이터에서 majority class로 collapse 하기 쉽다 (보고서 5.1 참조).
   실사용은 Stage 2(train_stage2_cot.py)의 CoT 학습을 권장.

실행:
    python train_stage1.py
"""
import os
import torch
from datasets import load_dataset

# Unsloth는 transformers/peft/trl 보다 먼저 import (최적화 패치 적용)
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


# ── 설정 ─────────────────────────────────────────────────────────────
MODEL_NAME     = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT   = False       # 일반 16-bit LoRA (24GB GPU). True 면 4-bit QLoRA(저VRAM)
DTYPE          = None        # 자동 (bf16 지원 시 bf16, 아니면 fp16)

HERE        = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE  = os.path.join(HERE, "data", "stage1_sample.jsonl")
OUTPUT_DIR  = os.path.join(HERE, "outputs", "bllossom_stage1")

LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05

PER_DEV_BATCH = 4
GRAD_ACCUM    = 4            # effective batch = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS    = 3
WARMUP_STEPS  = 5
LOGGING_STEPS = 1
SEED          = 42


# ── 1. 모델 + 토크나이저 ─────────────────────────────────────────────
print(f"\n[1/4] 모델 로드: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = DTYPE,
    load_in_4bit   = LOAD_IN_4BIT,
)

print(f"\n[2/4] LoRA adapter 부착 (r={LORA_R})")
model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_R,
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
    lora_alpha                 = LORA_ALPHA,
    lora_dropout               = LORA_DROPOUT,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = SEED,
    use_rslora                 = False,
)


# ── 2. 데이터셋 ──────────────────────────────────────────────────────
print(f"\n[3/4] 데이터셋 로드: {TRAIN_FILE}")
EOS = tokenizer.eos_token


def format_prompt(example):
    """AlphaFin instruction/input/output → Alpaca-style 단일 텍스트"""
    return {
        "text": (
            f"### 지시:\n{example['instruction']}\n\n"
            f"### 입력:\n{example['input']}\n\n"
            f"### 응답:\n{example['output']}{EOS}"
        )
    }


train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
train_ds = train_ds.map(format_prompt)
print(f"  train: {len(train_ds)}건")
print(f"\n샘플:\n{train_ds[0]['text'][:500]}...\n")


# ── 3. 학습 ──────────────────────────────────────────────────────────
print(f"[4/4] 학습 시작 → {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_ds,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LENGTH,
    packing            = False,
    args = TrainingArguments(
        per_device_train_batch_size = PER_DEV_BATCH,
        gradient_accumulation_steps = GRAD_ACCUM,
        warmup_steps                = WARMUP_STEPS,
        num_train_epochs            = NUM_EPOCHS,
        learning_rate               = LEARNING_RATE,
        bf16                        = torch.cuda.is_bf16_supported(),
        fp16                        = not torch.cuda.is_bf16_supported(),
        logging_steps               = LOGGING_STEPS,
        save_strategy               = "epoch",
        save_total_limit            = 1,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "cosine",
        seed                        = SEED,
        output_dir                  = OUTPUT_DIR,
        report_to                   = "none",
    ),
)

trainer.train()


# ── 4. 저장 ──────────────────────────────────────────────────────────
final_dir = os.path.join(OUTPUT_DIR, "final")
print(f"\n최종 모델(LoRA 어댑터) 저장: {final_dir}")
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)

# Stage 2(continue training)가 이어받을 merged 16-bit 모델 저장.
# 원본 AlphaFin 논문처럼 Stage 1 결과물 위에서 Stage 2를 학습하므로 필수.
merged_dir = os.path.join(OUTPUT_DIR, "merged_16bit")
print(f"merged 16-bit 저장 (Stage 2 이어학습용): {merged_dir}")
model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

print("\n✅ Stage 1 완료.")
print(f"→ 다음: python train_stage2_cot.py (이 결과물 {merged_dir} 를 이어받아 학습)")
