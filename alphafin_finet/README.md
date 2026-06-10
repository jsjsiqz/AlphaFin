# AlphaFin 한국판 — 학습 코드 샘플

AlphaFin 한국판 재현 프로젝트(KOSPI 삼성전자 005930)의 **2단계 순차 파인튜닝** 최소 예제입니다.
베이스 모델은 **Llama-3-Korean-Bllossom-8B**, 기법은 **LoRA + Unsloth** 입니다.

원본 논문처럼 **Stage 2는 Stage 1 결과물을 이어받아(continue training)** 학습합니다.

```
Bllossom-8B (base)
  └─ Stage 1 SFT (up/down 라벨)     → train_stage1.py
       └─ continue training (CoT)   → train_stage2_cot.py
```

- **Stage 1** (`train_stage1.py`): `output`이 `up`/`down` 라벨.
- **Stage 2** (`train_stage2_cot.py`): Stage 1 결과물을 이어받아 CoT(펀더멘털→기술적→종합→결론+확신도) 추론을 학습. 

## 데이터

`data/*.jsonl` — 각 라인은 `instruction` / `input` / `output` / `meta` 구조의 JSONL.

## 실행

```bash
# 의존성 (PyTorch는 CUDA 버전에 맞춰 먼저 설치)
pip install -r requirements.txt

# Stage 1 → 끝나면 merged_16bit 저장
python train_stage1.py

# Stage 2 → Stage 1 결과물을 이어받아 학습
python train_stage2_cot.py

# 예측 → Stage 2 결과물로 삼성전자 주가 방향 추론
python predict_stage2.py
```

> Stage 1을 먼저 돌려야 합니다. Stage 2는 Stage 1의 `merged_16bit`가 없으면 종료합니다.
> 이 샘플 데이터는 **파이프라인 검증용**이며, 실제 학습에는 전체 데이터셋을 사용하세요.
