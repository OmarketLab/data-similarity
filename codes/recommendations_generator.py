"""
============================================================================
 recommendations_generator.py  ―  추천 생성 (경량 버전, 파이썬 단독)
============================================================================
역할:
  임베딩 단계가 만든 상품 좌표 + 고객 구매목록 + 세그먼트 라벨을 받아서,
  고객마다 추천 상품 top-N(=3) 을 계산해 similarity_product_result.csv 로 떨군다.
  (경량 버전이라 pgvector 없이 numpy 로 코사인 유사도를 직접 계산)

추천 로직(WHY):
  - "고객을 세그먼트로 분류 -> 그 세그먼트 안에서 상품 유사도" 흐름을 그대로 구현.
  - 각 고객의 '취향 좌표(centroid)' = 그 고객이 산 상품들의 평균 좌표.
  - 추천 후보 = '같은 세그먼트 고객들이 산 상품' 풀로 한정(= 세그먼트 안에서).
    -> 그 세그먼트가 실제로 사는 상품들 중에서만 고르므로 맥락이 맞는 추천이 된다.
  - 점수 = 후보 좌표 · 취향좌표 (좌표가 단위벡터라 내적 = 코사인 유사도, 범위 -1~1).
  - 이미 산 상품은 제외(새 상품 추천이 목적).

match_reason(WHY):
  - 최종 산출물(similarity_product_result)의 match_reason(JSONB)은 그대로 Agent/Spring
    으로 전달되므로 API 명세서의 키 체계를 그대로 따른다.
      · co_purchased_with : 추천 상품과 임베딩상 가장 가까운 '이미 산 상품' 코드.
                            item2vec은 같은 장바구니 동시출현 상품끼리 가까워지므로
                            '임베딩 유사 = 함께 구매되는 경향' 으로 해석한다.
      · source            : 위 최근접 유사도가 임계값 이상이면 "동시구매"(특정 상품과의
                            연관이 추천을 끌었다고 보고 co_purchased_with 채움),
                            미만이면 "세그먼트선호"(세그먼트 전반 취향이 끌었다고 보고
                            co_purchased_with=null).
      · segment_support   : 그 세그먼트 고객 중 이 상품을 산 비율(0~1).

일관성(WHY):
  - 입력(좌표 파일 + 데이터)이 고정이고 numpy 연산이 결정적이므로,
    같은 입력이면 항상 같은 추천이 나온다. 결과를 CSV로 떨궈 그대로 DB에 박으면
    의도적으로 재생성하기 전까지 추천은 고정된다(= similarity_product_result 스냅샷).

필요 패키지: pip install pandas numpy
입력: data/output/product_vectors.npy, product_codes.json, embedding_meta.json (임베딩 산출물)
      data/output/customer_items.csv     (준비 단계 산출물)
      data/input/customer_result.csv     (이미 분류된 세그먼트: customer_id, segment, ...)
출력: data/output/similarity_product_result.csv
      (customer_id, stock_code, similarity_score, rank, match_reason)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 경로 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # codes/ -> 프로젝트 루트
INPUT = ROOT / "data" / "input"
OUTPUT = ROOT / "data" / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

TOP_N = 3                       # 고객당 추천 개수(= API Agent top_k 기본값과 동일 층).
SEG_COL = "segment"
CO_PURCHASE_THRESHOLD = 0.5     # 최근접 '이미 산 상품' 유사도가 이 값 이상이면 동시구매로 본다.

CUSTOMER_RESULT = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT / "customer_result.csv"

# ── 입력 로드 ────────────────────────────────────────────────────────────
vecs = np.load(OUTPUT / "product_vectors.npy")              # (P, 64) 단위벡터
codes = json.load(open(OUTPUT / "product_codes.json"))      # 길이 P
meta = json.load(open(OUTPUT / "embedding_meta.json"))
model_version = meta["model_version"]
code_idx = {c: i for i, c in enumerate(codes)}              # stock_code -> 행번호

cust_items = pd.read_csv(OUTPUT / "customer_items.csv",
                         dtype={"customer_id": str, "stock_code": str})  # customer_id, stock_code
seg = pd.read_csv(CUSTOMER_RESULT, dtype={"customer_id": str})[["customer_id", SEG_COL]]

# 임베딩이 있는 상품만 사용(임베딩 단계에서 min_count로 빠진 희소 상품 제외).
cust_items = cust_items[cust_items["stock_code"].isin(code_idx)]

# 고객 -> 세그먼트 매핑
cust2seg = dict(zip(seg["customer_id"], seg[SEG_COL]))

# (고객, 상품, 세그먼트) 결합 — 후보 풀과 segment_support 계산의 공통 베이스
tmp = cust_items.merge(seg, on="customer_id", how="inner")

# 세그먼트 -> 그 세그먼트가 산 상품(행번호 배열)  : 추천 '후보 풀'
seg_pool = {}
for s, g in tmp.groupby(SEG_COL):
    seg_pool[s] = np.array(sorted({code_idx[c] for c in g["stock_code"]}))

# segment_support: 세그먼트별 (이 상품을 산 고객 수 / 세그먼트 전체 고객 수)
seg_cust_count = seg.groupby(SEG_COL)["customer_id"].nunique().to_dict()
seg_support = {}                                            # seg_support[segment][stock_code] = 비율(0~1)
for (s, code), n_buyers in tmp.groupby([SEG_COL, "stock_code"])["customer_id"].nunique().items():
    seg_support.setdefault(s, {})[code] = n_buyers / seg_cust_count[s]

# 고객 -> 산 상품(행번호 집합)
bought_by_cust = (
    cust_items.groupby("customer_id")["stock_code"]
              .apply(lambda s: np.array([code_idx[c] for c in s]))
              .to_dict()
)

# ── 고객별 추천 계산 ─────────────────────────────────────────────────────
rows = []
for cid, bought_idx in bought_by_cust.items():
    s = cust2seg.get(cid)
    if s is None or s not in seg_pool:
        continue                                  # 세그먼트 정보 없는 고객은 건너뜀

    # 취향 좌표 = 산 상품 좌표들의 평균 -> 다시 단위벡터로 정규화
    centroid = vecs[bought_idx].mean(axis=0)
    n = np.linalg.norm(centroid)
    if n == 0:
        continue                                  # (이론상) 좌표가 0이면 건너뜀
    centroid /= n

    # 후보 = 같은 세그먼트 풀에서 '이미 산 상품' 제외
    cand = np.setdiff1d(seg_pool[s], bought_idx, assume_unique=False)
    if cand.size == 0:
        continue

    # 점수 = 후보좌표 · 취향좌표 (단위벡터 내적 = 코사인 유사도)
    scores = vecs[cand] @ centroid
    # 상위 TOP_N 만 추리기 (argpartition 으로 빠르게 후 정렬)
    k = min(TOP_N, cand.size)
    top = cand[np.argpartition(-scores, k - 1)[:k]]
    top = top[np.argsort(-(vecs[top] @ centroid))]   # 최종 내림차순 정렬

    bought_vecs = vecs[bought_idx]                   # (B, 64) 이 고객이 산 상품 좌표
    support_map = seg_support.get(s, {})

    for rank, p in enumerate(top, start=1):
        score = round(float(vecs[p] @ centroid), 6)

        # match_reason — 추천 상품 p 와 가장 가까운 '이미 산 상품' 찾기
        sim_to_bought = bought_vecs @ vecs[p]        # (B,)
        j = int(np.argmax(sim_to_bought))
        nearest_sim = float(sim_to_bought[j])
        if nearest_sim >= CO_PURCHASE_THRESHOLD:
            source = "동시구매"
            co_purchased_with = codes[bought_idx[j]]
        else:
            source = "세그먼트선호"
            co_purchased_with = None
        match_reason = {
            "co_purchased_with": co_purchased_with,
            "source": source,
            "segment_support": round(support_map.get(codes[p], 0.0), 6),
        }

        rows.append((cid, codes[p], score, rank,
                     json.dumps(match_reason, ensure_ascii=False)))

out = pd.DataFrame(rows, columns=["customer_id", "stock_code", "similarity_score", "rank", "match_reason"])
out.to_csv(OUTPUT / "similarity_product_result.csv", index=False)

print(f"[추천] 생성 완료: 고객 {out['customer_id'].nunique():,}명 / 행 {len(out):,}개 "
      f"-> {OUTPUT / 'similarity_product_result.csv'}")
print(f"[추천] model_version = {model_version} (컬럼에서는 제외, 추적용 로그)")
