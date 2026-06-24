"""
============================================================================
 embedding_training.py  ―  item2vec 임베딩 학습 (경량 버전)
============================================================================
역할:
  baskets.txt 를 word2vec(=item2vec)으로 학습해서, 각 상품코드를 64차원 좌표로
  바꾼 뒤 '파일'로 저장한다. (경량 버전이라 DB가 아니라 파일에 영속화)

핵심 설계 의도(WHY):
  1) "왜 word2vec 인가"
     한 주문(바스켓)을 문장, 상품코드를 단어로 보고 학습하면
     '자주 함께 담기는 상품'끼리 좌표가 가까워진다 = 구매행동(함께 사는) 유사도.
  2) "왜 한 번 학습해서 파일로 저장하는가" (★ 일관성의 핵심)
     word2vec 은 무작위 초기화/네거티브 샘플링이 있어 돌릴 때마다 결과가 달라진다.
     매 요청마다 재학습하면 추천이 흔들린다. 그래서 '한 번 학습 -> 파일 고정 ->
     이후엔 그 파일만 사용' 으로 일관성을 보장한다.
  3) "왜 seed 고정 + workers=1 인가" (★ 재현성)
     멀티스레드(workers>1)는 학습 순서가 비결정적이라 seed 를 줘도 결과가 매번
     달라진다. 완전한 재현을 위해 단일 스레드로 학습한다(느리지만 결정적).
     추가로 PYTHONHASHSEED=0 환경변수까지 고정하면 해시까지 결정적이 된다.
        예) PYTHONHASHSEED=0 python codes/embedding_training.py
  4) "왜 L2 정규화 하는가"
     좌표를 단위벡터(길이 1)로 만들면, 내적(dot)이 곧 코사인 유사도가 되어
     추천 단계의 계산이 단순/안정적이 된다.

필요 패키지: pip install gensim pandas numpy
입력: data/output/baskets.txt              (준비 단계 산출물)
출력: data/output/product_vectors.npy      (N x 64 float32, L2 정규화됨)
      data/output/product_codes.json       (행 순서에 대응하는 stock_code 목록)
      data/output/embedding_meta.json       (model_version 등 메타 — 재학습 추적용)
"""
import json
import datetime
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# ── 경로 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # codes/ -> 프로젝트 루트
OUTPUT = ROOT / "data" / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)


class EpochLogger(CallbackAny2Vec):
    def __init__(self, total):
        self.epoch, self.total = 0, total
    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"  학습 진행: {self.epoch}/{self.total} epoch 완료")

# ── 하이퍼파라미터 (값마다 이유를 주석으로) ──────────────────────────────
DIM = 64          # 임베딩 차원. 데이터가 '넉넉히 작은' 규모(토큰 ~76만)라
                  # 64면 충분하고, 너무 키우면(예:300) 희소 상품을 과적합한다.
WINDOW = 999      # 바스켓 안에서는 상품 '순서'가 의미 없으므로, 윈도우를 크게 잡아
                  # 한 주문 전체를 한 상품의 문맥으로 본다(= 같은 주문이면 다 이웃).
MIN_COUNT = 5     # 5회 미만 등장 상품은 좌표가 불안정 -> 제외. (전체 상품의 약 8%,
                  # 토큰의 0.1%만 빠지므로 손실은 거의 없고 잡음만 제거된다.)
SG = 1            # skip-gram. 동시구매처럼 '드문 조합'까지 잡는 데 CBOW보다 유리.
NEGATIVE = 10     # 네거티브 샘플링 수. 소규모 코퍼스에서 안정적인 기본값.
EPOCHS = 30       # 코퍼스가 작아 여러 번 돌려야 충분히 학습된다.
SEED = 42         # 재현성용 고정 시드.

# model_version: 재학습할 때마다 바뀌는 식별자. 추천/DB에서 "어느 학습본이 만든
# 결과인지" 추적해, 나중에 데이터가 늘어 재학습할 때 추천 churn 을 관리하기 위함.
MODEL_VERSION = "item2vec_v1_" + datetime.date.today().strftime("%Y%m%d")

# ── 학습 ────────────────────────────────────────────────────────────────
baskets = [line.split() for line in open(OUTPUT / "baskets.txt")]

model = Word2Vec(
    sentences=baskets,
    vector_size=DIM,
    window=WINDOW,
    min_count=MIN_COUNT,
    sg=SG,
    negative=NEGATIVE,
    epochs=EPOCHS,
    workers=1,        # ★ 결정성을 위해 단일 스레드 (위 WHY 3 참고)
    seed=SEED,
    callbacks=[EpochLogger(EPOCHS)],
)

# ── 좌표 추출 + L2 정규화 + 파일 저장 ────────────────────────────────────
codes = list(model.wv.index_to_key)                 # 학습된 상품코드(빈도순)
vecs = model.wv[codes].astype(np.float32)
vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)   # 단위벡터로 정규화

np.save(OUTPUT / "product_vectors.npy", vecs)               # 좌표 행렬 (행 순서 = codes 순서)
json.dump(codes, open(OUTPUT / "product_codes.json", "w"))  # 행↔stock_code 매핑
json.dump(
    {"model_version": MODEL_VERSION, "dim": DIM, "n_products": len(codes),
     "min_count": MIN_COUNT, "epochs": EPOCHS, "seed": SEED},
    open(OUTPUT / "embedding_meta.json", "w"), ensure_ascii=False, indent=2,
)

print(f"[임베딩] 학습 완료: 상품 {len(codes):,}종 x {DIM}차원 -> {OUTPUT}")
print(f"[임베딩] model_version = {MODEL_VERSION}")
