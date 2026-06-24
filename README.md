# data-similarity
데이터 유사도 검색 프로세스


## 파일 실행 순서  
> `data_preparing.py` → `embedding_training.py` → `recommendations_generator.py` → `db_loader.py`  


## 사전 준비
1. 패키지  
`pip install gensim pandas numpy`  
  
2. 데이터  
`online_retail_II_cleaned.csv`(전처리 데이터)  
`customer_result.csv`(분류 완료 데이터)
  
3. PostgreSQL  (MVP x)
`psql`로 접속할 PostgreSQL DB.

## 실행
### 1. 데이터 준비
```bash
python data_preparing.py online_retail_II_cleaned.csv
```
- baskets.txt (학습용 코퍼스)
- customer_items.csv (고객별 구매목록)  

### 2. item2vec 임베딩 학습
```bash
$env:PYTHONHASHSEED=0
python embedding_training.py
```
```bash
$env:PYTHONHASHSEED=0; python embedding_training.py
```
- `PYTHONHASHSEED=0` 을 붙이는 이유: 해시까지 고정해 매번 **같은 결과(재현성)** 를 보장하기 위함. (스크립트 안에서도 seed 고정 + 단일 스레드로 학습)
- 산출: `product_vectors.npy`, `product_codes.json`, `embedding_meta.json`(model_version 등)
- 학습은 한 번만 하면 됩니다. 이 파일들이 곧 "고정된 상품 좌표"이고, 이후 추천은 항상 이걸 재사용 → 추천이 흔들리지 않습니다.  

### 3. 추천 생성
```bash
python recommendations_generator.py customer_result.csv
```
- 각 고객의 취향 좌표(=산 상품들의 평균)를 구해, **같은 세그먼트가 사는 상품 풀** 안에서 가까운 미구매 상품 top-10을 뽑습니다.
- 산출: `recommend_products.csv` (컬럼: `customer_id, stock_code, similarity_score, rank, model_version`)  

### 4. DB 적재   (MVP x)

프로젝트 루트에서 테이블을 만들고:
```bash
bashpsql -d <DB이름> -f code/db_loader.sql
```
같은 루트에서 psql 을 띄워 CSV 적재(경로가 루트 기준이라 그대로 동작):
```bash
bashpsql -d <DB이름>
```
```sql
\copy recommend_products(customer_id, stock_code, similarity_score, rank, model_version) FROM 'data/output/recommend_products.csv' WITH (FORMAT csv, HEADER true);
```

