# Mission 18 — 영화 리뷰 감성 분석 웹앱
이 서비스는 사용자가 영화 정보를 등록하고, 해당 영화에 대한 리뷰를 작성하면 **리뷰 감성 분석**을 수행하여 긍정/중립/부정 레이블과 감성 점수를 시각적으로 보여줍니다.

모든 데이터는 **FastAPI 백엔드에서 관리**되며, 프론트엔드(Streamlit)는 API를 통해 데이터를 주고받습니다.


## 시스템 아키텍처
```
Mission18
├── .streamlit
│ └── secrets.toml # Streamlit 환경 변수 (BACKEND_URL 등)
│
├── backend (FastAPI)
│ ├── app.py # 메인 백엔드 API (영화, 리뷰, 감성분석)
│ ├── db.py # SQLite DB 초기화 및 세션 관리
│ ├── models.py # SQLModel ORM (Movie, Review)
│ ├── sentiment.py # 감성 분석 (PyTorch XLM-R + 동적 양자화)
│ └── requirements.txt
│
├── frontend (Streamlit)
│ ├── app.py # 웹 UI: 영화 목록, 등록, 리뷰 작성, 감성 결과 표시
│ └── requirements.txt
│
├── movies.db # SQLite 데이터베이스
├── pyproject.toml / uv.lock # uv 가상환경 및 의존성 관리
└── README.md # 프로젝트 문서
```
```
# .streamlit/secrets.toml.example
BACKEND_URL = "http://127.0.0.1:8000"
```

---

## 기술 스택

| 구분 | 사용 기술 |
|------|-------------|
| **Frontend** | Streamlit |
| **Backend** | FastAPI, SQLModel |
| **Database** | SQLite |
| **ML Model** | XLM-RoBERTa (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) |
| **Optimization** | PyTorch Dynamic Quantization (INT8) |
| **Deploy (optional)** | Streamlit Cloud, Render, AWS EC2 등 |


---

## 빠른 실행
### 1) 백엔드
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
# http://127.0.0.1:8000/docs
```

### 2) 프론트엔드
```bash
cd ../frontend
uv pip install -r requirements.txt
uv run streamlit run app.py
```

---

## 주요 기능
### 영화 관리
- 영화 등록 (제목, 감독, 장르, 개봉일, 포스터 URL)
- 영화 목록 조회
- 영화 삭제

### 리뷰 관리
- 영화별 리뷰 작성
- 감성 분석 자동 수행
- 최근 10개 리뷰 표시

### 감성 분석 (Sentiment Analysis)
- 모델: cardiffnlp/twitter-xlm-roberta-base-sentiment
- 가속화: PyTorch 동적 양자화 (CPU 환경에서 약 1.5~2x 속도 향상)
- 출력:
    - ```label```: positive / neutral / negative
    - ```score```: [-1, +1] 범위 (pos_prob - neg_prob)

---

## 백엔드 주요 API
| 메서드      | 엔드포인트                         | 설명                 |
| -------- | ----------------------------- | ------------------ |
| `POST`   | `/movies`                     | 영화 등록              |
| `GET`    | `/movies`                     | 영화 목록 조회           |
| `GET`    | `/movies/{id}`                | 특정 영화 조회           |
| `DELETE` | `/movies/{id}`                | 영화 삭제              |
| `POST`   | `/reviews`                    | 리뷰 등록 + 감성 분석 수행   |
| `GET`    | `/reviews?movie_id={id}`      | 영화별 리뷰 조회          |
| `GET`    | `/ratings/{movie_id}`         | 평균 감성 점수 조회        |
| `GET`    | `/sentiment/analyze?text=...` | 단일 문장 감성 분석 (테스트용) |


---

## ERD (데이터베이스 구조)
```
┌───────────────┐       ┌────────────────┐
│   Movies      │1     ∞│    Reviews     │
├───────────────┤       ├────────────────┤
│ id (PK)       │◀──────│ movie_id (FK)  │
│ title         │       │ reviewer       │
│ director      │       │ content        │
│ genre         │       │ sentiment      │
│ release_date  │       │ score          │
│ poster_url    │       │ created_at     │
└───────────────┘       └────────────────┘
```

---

## 모델 최적화 방식
| 구분 | 설명                             |
| -- | ------------------------------ |
| 방식 | PyTorch Dynamic Quantization   |
| 대상 | Linear 레이어                     |
| 효과 | CPU 연산량 및 메모리 사용량 감소           |
| 이유 | Windows 환경에서도 GPU 없이 안정적 추론 가능 |

---

## 개선 아이디어
- ONNX Runtime 기반 최적화로 전환 (향후 계획)
- 한국어 전용 모델 교체 (예: KcELECTRA, KoBERT)
- 감성 점수 기반 평점 자동화 기능
- 영화 포스터 이미지 + 감정 비율 시각화

----
## 라이선스
MIT License
© 2025 FinneKim.All Rights Reserved.
