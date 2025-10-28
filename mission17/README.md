# Mission 17 - MNIST 손글씨 인식 서비스 (ONNX + Streamlit + Docker)

본 프로젝트는 ONNX Runtime을 활용하여 MNIST 손글씨 숫자 분류 모델을 웹 애플리케이션 형태로 구현한 사례입니다.  
Streamlit 기반의 인터페이스를 제공하며, Docker 환경에서 손쉽게 배포할 수 있습니다.

---

## 1. 프로젝트 개요

| 구분 | 내용 |
|------|------|
| **프로젝트명** | Mission 17 - MNIST 손글씨 인식 |
| **주요 목적** | 사용자가 직접 그린 숫자를 ONNX 모델로 분류하는 AI 서비스 구현 |
| **모델** | MNIST ONNX 모델 ([ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist/model)) |
| **추론 엔진** | onnxruntime |
| **웹 프레임워크** | Streamlit |
| **배포 방식** | Docker / Docker Compose |
| **Docker Hub URL** | [https://hub.docker.com/r/fiinn/mission17-mnist](https://hub.docker.com/r/fiinn/mission17-mnist) |

---

## 2. 시스템 구성도
```
사용자 입력 (Canvas)
↓
이미지 전처리 (28x28, 흑백 변환, 정규화)
↓
ONNX Runtime 추론 (MNIST 모델)
↓
결과 시각화 (확률 차트 + 예측 숫자)
↓
히스토리 저장 (세션 관리)
```

---

## 3. 디렉토리 구조
```
mission17/
├── app.py # 모델 로드 및 예측 로직
├── ui.py # Streamlit 사용자 인터페이스
├── Dockerfile # Docker 이미지 빌드 설정
├── docker-compose.yml # 컨테이너 실행 환경 정의
├── requirements.txt # Python 패키지 의존성 목록
├── .dockerignore # Docker 빌드시 제외할 파일
└── README.md # 프로젝트 문서
```

---

## 4. 주요 기능

### (1) 사용자 입력
- Streamlit의 `streamlit-drawable-canvas`를 사용하여 마우스로 직접 숫자를 입력할 수 있는 캔버스를 제공

### (2) 이미지 전처리
- 280×280 RGBA 이미지를 28×28 흑백 이미지로 변환
- 배경 반전 및 정규화 수행하여 ONNX 모델 입력 형태에 맞춤

### (3) 모델 추론
- ONNX Runtime을 이용하여 추론을 수행하고, softmax를 통해 0~9 클래스 확률 산출

### (4) 결과 시각화
- Streamlit의 bar_chart를 사용해 예측 확률을 시각적으로 표시
- 상위 3개 확률과 예측 라벨을 함께 표시

### (5) 이미지 히스토리
- 사용자가 예측한 이미지를 세션 상태(`st.session_state`)에 저장하여 직전 예측 내역을 유지
- 원본 이미지와 전처리된 이미지를 모두 썸네일 형태로 저장 및 표시

---

## 5. 실행 방법

### 5.1 로컬 환경에서 실행

```
# 가상환경 생성 및 패키지 설치
python -m venv .venv
source .venv/Scripts/activate        # (Windows)
pip install -r requirements.txt


# Streamlit 실행
python run streamlit run ui.py
브라우저에서 다음 주소로 접속합니다.
http://localhost:8501
```

### 5.2 Docker 환경에서 실행
(1) Docker 단독 실행
```
docker build -t mission17-mnist .
docker run -p 8501:8501 mission17-mnist
```

(2) Docker Compose 실행
```
docker compose up --build
```
이후 웹 브라우저에서 http://localhost:8501 접속 시 서비스가 실행됩니다.

---


## 6. 사용 방법
1. 캔버스에 0~9 사이의 숫자를 마우스로 입력합니다.
2. “예측하기” 버튼을 클릭합니다.
3. 모델이 숫자를 인식하여 확률 분포(0~9)를 시각화합니다.
4. 히스토리 영역에서 과거 입력 및 예측 결과를 확인할 수 있습니다.

---


## 7. 시연 화면 예시
아래 이미지는 실제 Streamlit 실행 화면을 예시로 제공합니다.
(캡처 이미지를 screenshot.png로 프로젝트 폴더에 추가할 수 있습니다.)

```
+------------------+------------------+
| 입력 캔버스      | 전처리 이미지     |
+------------------+------------------+
| 예측 확률 차트   | 이미지 히스토리   |
+------------------+------------------+
```

---

## 8. 참고 자료
[ONNX Model Zoo - MNIST 모델](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist/model?utm_source=chatgpt.com)

[Streamlit 공식 문서](https://docs.streamlit.io/)

[ONNX Runtime 공식 문서](https://onnxruntime.ai/)