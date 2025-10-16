# 학습 성취도 예측 모델 (Performance Index Prediction)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://hub.docker.com/repository/docker/fiinn/mission15-r1)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 프로젝트 개요
본 프로젝트는 학생들의 학습 패턴 데이터를 활용해 **Performance Index(성취 지수)** 를 예측하는 머신러닝 회귀 모델을 개발하는 미션입니다.  
데이터 전처리부터 모델 학습, 성능 평가, Docker 환경 구성까지 **엔드투엔드 ML 파이프라인**을 구성했습니다.

---

## 데이터 구성

| 변수명 | 설명 |
|:--------|:------|
| **Hours Studied** | 학생이 공부에 투자한 총 시간 |
| **Previous Scores** | 이전 시험 점수 |
| **Extracurricular Activities** | 과외 활동 참여 여부 (Yes / No) |
| **Sleep Hours** | 하루 평균 수면 시간 |
| **Sample Question Papers Practiced** | 연습한 모의고사 수 |
| **Performance Index** | 예측 목표 변수 (학습 성취도 지표) |

---

## 기술 스택

| 구분 | 기술 / 도구 |
|------|--------------|
| **언어** | Python 3.9 |
| **라이브러리** | pandas, numpy, matplotlib, scikit-learn |
| **모델** | Random Forest Regressor |
| **환경 관리** | Docker, Jupyter Notebook |
| **버전 관리** | Git / GitHub |

---

## 모델링 요약

- **전처리 파이프라인**
  - 수치형 변수 → `StandardScaler`
  - 범주형 변수 → `OneHotEncoder`
- **학습 모델**
  - `RandomForestRegressor(n_estimators=200, random_state=42)`
- **성능 지표**
  - R² ≈ **0.92**
  - RMSE ≈ **1.8**
- **특징 중요도**
  - Previous Scores > Hours Studied > Sample Question Papers Practiced

---

## 프로젝트 구조

```
mission15/
│
├── mission-result/
│   ├── Mission15_Report.pdf          # 최종 보고서 (한글 폰트 포함)
│   ├── mission15_train.csv           # 학습 데이터
│   ├── mission15_test.csv            # 테스트 데이터
│   ├── mission15_notebook.ipynb      # 분석 및 모델링 코드
│   ├── researcher1_result.png        # 시각화 이미지
│   └── researcher2_result.png
│
└── README.md                         # 본 문서
```

---

## Docker 환경 구성

```bash
# Docker 이미지 빌드
docker build -t fiinn/mission15-r1 .

# 컨테이너 실행
docker run -p 10000:8888 fiinn/mission15-r1
```

| [Docker Hub에서 체크](https://hub.docker.com/repository/docker/fiinn/mission15-r1/general)
