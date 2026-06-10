"""
[샘플] Llama-3-Korean-Bllossom-8B + CoT QLoRA fine-tuning — Stage 2 (이어학습)

AlphaFin 한국판 재현 프로젝트의 Stage 2 학습 코드 (GitHub 공개용 최소 예제).
output = 4단계 추론 ~750자 (펀더멘털 → 기술적 → 종합 판단 → 결론+확신도).

★ 원본 AlphaFin 논문 방식(continue training): Stage 2는 base가 아니라
  **Stage 1 결과물(outputs/bllossom_stage1/merged_16bit)을 이어받아** 학습한다.
  → 따라서 반드시 `python train_stage1.py` 를 먼저 실행해야 한다.

      ChatGLM2/Bllossom (base)
        │ Stage 1 SFT (up/down)
        ▼
      Stage 1 결과물  ──(이 스크립트가 이어받음)──┐
        │ continue training (CoT)                │
        ▼                                         ◀
      Stage 2 결과물

Stage 1 과의 차이:
  - 시작 모델:        base ✕ → Stage 1 결과물 ✔ (continue training)
  - 데이터:           sample/data/stage2_sample.jsonl  (CoT 출력, 최신 2건 샘플)
  - MAX_SEQ_LENGTH:   3072 (CoT 입력+출력이 ~2666 토큰 → 2048이면 응답이 잘림)
  - response_template: 응답 토큰만 loss 계산 → collapse / empty-output 방지
  - epoch 3 → 5, LR 2e-4 → 5e-5 (긴 출력은 천천히), weight_decay 0.01 → 0.05

DataCollatorForCompletionOnlyLM 은 trl 버전에 따라 import 경로가 달라
3중 fallback (top-level → submodule → inline) 을 적용한다.

  - 타겟 GPU: RTX 3090 / 4090 / A6000 (24GB)

실행:
    python train_stage1.py                       # 먼저 Stage 1
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    python train_stage2_cot.py                   # 그 결과물을 이어받아 Stage 2
"""
import os
import torch
from datasets import load_dataset

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling


# ── DataCollatorForCompletionOnlyLM import (trl 버전 호환 3중 fallback) ─
try:
    from trl import DataCollatorForCompletionOnlyLM
    print("[collator] trl 상위에서 import")
except ImportError:
    try:
        from trl.trainer.utils import DataCollatorForCompletionOnlyLM
        print("[collator] trl.trainer.utils 에서 import")
    except ImportError:
        print("[collator] inline 정의 사용 (trl에 없음)")

        class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
            """응답 부분만 loss 계산 (앞쪽 prompt 토큰은 -100 마스킹)"""
            def __init__(self, response_template, tokenizer, **kwargs):
                super().__init__(tokenizer=tokenizer, mlm=False, **kwargs)
                if isinstance(response_template, str):
                    self.response_template_ids = tokenizer.encode(
                        response_template, add_special_tokens=False
                    )
                else:
                    self.response_template_ids = response_template

            def torch_call(self, examples):
                batch = super().torch_call(examples)
                tmpl = self.response_template_ids
                for i, ids in enumerate(batch["input_ids"]):
                    ids_list = ids.tolist()
                    start = -1
                    for j in range(len(ids_list) - len(tmpl) + 1):
                        if ids_list[j : j + len(tmpl)] == tmpl:
                            start = j + len(tmpl)
                            break
                    if start > 0:
                        batch["labels"][i, :start] = -100
                    else:
                        batch["labels"][i, :] = -100  # 응답 못 찾으면 전부 마스킹 (안전)
                return batch


# ── 설정 ─────────────────────────────────────────────────────────────
HERE           = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL     = "MLP-KTLim/llama-3-Korean-Bllossom-8B"   # 참고: Stage 1의 출발점
STAGE1_DIR     = os.path.join(HERE, "outputs", "bllossom_stage1", "merged_16bit")
MODEL_NAME     = STAGE1_DIR     # ★ Stage 1 결과물에서 이어서 학습 (continue training)

MAX_SEQ_LENGTH = 3072   # 샘플 CoT 데이터가 ~2666 토큰 → 2048이면 응답이 잘려 loss=0
LOAD_IN_4BIT   = False  # 일반 16-bit LoRA (24GB GPU). True 면 4-bit QLoRA(저VRAM)
DTYPE          = None   # 자동 (bf16 지원 시 bf16, 아니면 fp16)

TRAIN_FILE  = os.path.join(HERE, "data", "stage2_sample.jsonl")
OUTPUT_DIR  = os.path.join(HERE, "outputs", "bllossom_stage2_cot")

LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05

PER_DEV_BATCH = 1
GRAD_ACCUM    = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS    = 5
WARMUP_STEPS  = 5
LOGGING_STEPS = 1
SEED          = 42

RESPONSE_TEMPLATE = "### 응답:\n"


# ── 1. 모델 + 토크나이저 (Stage 1 결과물에서 이어받기) ───────────────
if not os.path.isdir(MODEL_NAME):
    raise SystemExit(
        f"\n✗ Stage 1 결과물을 찾을 수 없습니다: {MODEL_NAME}\n"
        f"  Stage 2는 Stage 1을 이어받아 학습합니다(continue training).\n"
        f"  → 먼저 `python train_stage1.py` 를 실행하세요 "
        f"(merged_16bit 가 저장됩니다).\n"
    )

print(f"\n[1/4] 모델 로드 (continue training): {MODEL_NAME}")
print(f"      (base={BASE_MODEL} 에서 Stage 1 학습된 결과물)")
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
    return {
        "text": (
            f"### 지시:\n{example['instruction']}\n\n"
            f"### 입력:\n{example['input']}\n\n"
            f"### 응답:\n{example['output']}{EOS}"
        )
    }


train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
train_ds = train_ds.map(format_prompt)
print(f"  train (CoT): {len(train_ds)}건")

sample_lens = [len(tokenizer(x["text"])["input_ids"]) for x in train_ds]
print(f"  토큰 길이: min={min(sample_lens)}, max={max(sample_lens)}, "
      f"avg={sum(sample_lens)//len(sample_lens)}")
if max(sample_lens) > MAX_SEQ_LENGTH:
    print(f"  ⚠ MAX_SEQ_LENGTH({MAX_SEQ_LENGTH}) 초과 → truncate됨")

print(f"\n샘플 (앞 600자):\n{train_ds[0]['text'][:600]}...\n")


# ── 3. 학습 (응답만 loss) ────────────────────────────────────────────
print(f"[4/4] 학습 시작 → {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

collator = DataCollatorForCompletionOnlyLM(
    response_template = RESPONSE_TEMPLATE,
    tokenizer         = tokenizer,
)

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_ds,
    data_collator      = collator,
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
        weight_decay                = 0.05,
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

# Stage 1 과 동일하게 merged 16-bit 모델도 별도 저장 (배포/평가/추가 이어학습용).
merged_dir = os.path.join(OUTPUT_DIR, "merged_16bit")
print(f"merged 16-bit 저장 (별도): {merged_dir}")
model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

print("\n✅ Stage 2 (CoT) 완료.")
print(f"평가: python evaluate.py --model {final_dir} --max-new 600")
