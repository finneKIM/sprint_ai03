# 베이스 이미지: 가벼운 Debian slim + Python 3.10
FROM python:3.10-slim

# 런타임 품질/속도 관련 기본 환경
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 작업 디렉토리
WORKDIR /app

# 1) 시스템 패키지 + 파이썬 의존성 설치
# - libgomp1: onnxruntime가 사용하는 OpenMP 런타임 (없으면 런타임 에러 날 수 있음)
# - curl/ca-certificates: 모델 다운로드용
COPY requirements.txt ./requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates libgomp1 && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# 2) 모델 미리 다운로드(네트워크 없는 환경에서도 실행되도록)
RUN mkdir -p models && \
    curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx" \
      -o models/mnist.onnx

# 3) 앱 소스 복사
COPY . .

# 4) 포트 노출
EXPOSE 8501

# 5) 실행 커맨드 (Streamlit)
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
