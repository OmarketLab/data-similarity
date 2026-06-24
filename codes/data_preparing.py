"""
============================================================================
 data_preparing.py  ―  데이터 준비 (경량 버전)
============================================================================
역할:
  거래 CSV 한 장을 받아서, 뒤 단계가 쓸 두 가지 산출물을 만든다.
    (1) baskets.txt          : item2vec 학습용 코퍼스 (한 줄 = 한 주문)
    (2) customer_items.csv   : 고객별 구매 상품 목록 (추천 생성용)
                               * 경량 파이프라인이라 추가 의존성(parquet 엔진) 없이
                                 어디서나 읽히는 CSV 로 떨군다.

설계 의도(WHY):
  - 경량 버전이라 DB(pgvector)를 쓰지 않고, 임베딩/추천 계산을 전부 파이썬에서
    하고 '최종 추천 결과'만 DB(similarity_product_result)로 보낸다.
    -> 그래서 여기서는 DB 적재용 포맷이 아니라, 파이썬 계산에 바로 쓰기 좋은
       중간 산출물(텍스트 코퍼스 / csv)로 떨군다.
  - country 컬럼: 분석에 쓰지 않기로 했으므로 즉시 제거.
  - description 컬럼: item2vec은 '상품명 텍스트'를 전혀 쓰지 않고
    '같은 주문에 어떤 상품코드들이 함께 담겼는가'만 본다. 따라서 임베딩 파이프라인
    에서는 description이 불필요 -> 제거. (상품명은 DB invoice_line 에만 보관)

경로 규약(WHY):
  - 코드는 codes/ 아래, 데이터는 data/input · data/output 아래로 분리한다.
  - 어느 위치에서 실행하든 동작하도록 스크립트 파일 기준 절대경로(ROOT)를 잡는다.
입력: data/input/online_retail_II_cleaned.csv   (인자로 다른 경로 지정 가능)
출력: data/output/baskets.txt
      data/output/customer_items.csv
"""
import sys
from pathlib import Path

import pandas as pd

# ── 경로 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # codes/ -> 프로젝트 루트
INPUT = ROOT / "data" / "input"
OUTPUT = ROOT / "data" / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)               # 산출물 폴더 보장

# 입력 CSV 경로(실제 운영 입력은 csv). 인자로 받거나 기본값 사용.
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT / "online_retail_II_cleaned.csv"

# dtype 지정 이유: stock_code 가 '79323P' 처럼 문자+숫자라서 자동추론되면
# 숫자형 코드가 int로 바뀌어 '85048' vs 85048 처럼 키가 어긋난다. 처음부터 문자열 고정.
df = pd.read_csv(SRC, dtype={"stock_code": str, "customer_id": str, "invoice_id": str})

# --- 불필요 컬럼 제거 (있을 때만) ---
# errors='ignore': country/description 가 이미 없는 CSV가 들어와도 깨지지 않게.
df = df.drop(columns=["country", "description"], errors="ignore")

# ---------------------------------------------------------------------------
# (1) 바스켓 코퍼스
#     "한 주문 = 한 문장", "상품코드 = 단어" 로 보고 item2vec 을 학습시킬 것이므로
#     주문별 상품코드 목록을 만든다.
# ---------------------------------------------------------------------------
baskets = (
    df.sort_values("line_id")                       # 주문 내 등장 순서 보존(가독성 목적)
      .groupby("invoice_id")["stock_code"]
      .apply(lambda s: list(dict.fromkeys(s)))       # 한 주문 안 중복 상품은 1번만(집합 의미)
)
# 상품이 1종뿐인 주문은 '함께 담김' 신호가 0이라 학습에 무의미 -> 제외.
baskets = baskets[baskets.map(len) >= 2]

with open(OUTPUT / "baskets.txt", "w") as f:
    for items in baskets:
        f.write(" ".join(items) + "\n")

# ---------------------------------------------------------------------------
# (2) 고객별 구매 상품 목록
#     추천 단계에서 "이 고객이 산 상품들의 평균 좌표"를 구해야 하므로,
#     (고객, 상품) 구매 쌍을 중복 없이 떨군다.
# ---------------------------------------------------------------------------
cust_items = df[["customer_id", "stock_code"]].dropna().drop_duplicates()
cust_items.to_csv(OUTPUT / "customer_items.csv", index=False)

print(f"[준비] 바스켓 {len(baskets):,}건 -> {OUTPUT / 'baskets.txt'}")
print(f"[준비] 고객-상품 구매쌍 {len(cust_items):,}건 -> {OUTPUT / 'customer_items.csv'}")
