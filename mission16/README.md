# Mission 16: 딥러닝 모델 양자화 및 추론 최적화

본 문서는 AI Engineer Training 3기에서 수행한 Mission 16 프로젝트에 대한 요약 보고서임.  
모델 경량화 및 성능 최적화를 목적으로 양자화 기법을 적용하고, PyTorch 및 ONNX 기반 추론 환경에서 성능을 비교 검증함.

---

## 1. 프로젝트 개요

- 주제: 딥러닝 모델 양자화 및 추론 최적화  
- 목적: 모델 크기 축소와 연산 효율 향상을 통해 추론 속도 개선  
- 수행 환경:
  - Python 3.9 이상
  - PyTorch, ONNX Runtime 기반
  - Ubuntu 22.04 환경 기준

---

## 2. 주요 기술 및 방법론

- **Post-Training Quantization (PTQ)**  
  학습 완료된 모델을 기반으로 양자화 수행하여 크기 감소 및 계산 효율 향상  
- **Quantization-Aware Training (QAT)**  
  학습 단계에서 양자화 효과를 반영하여 정밀도 손실 최소화  
- **대칭(symmetric) / 비대칭(asymmetric) quantization 비교**  
  스케일(scale)과 제로 포인트(zero-point) 계산 방식 차이에 따른 성능 영향 분석  
- **ONNX 변환 및 추론 검증**  
  모델을 ONNX 포맷으로 변환하여 다양한 환경에서의 추론 가능성 확인

---

## 3. 디렉터리 구조
```
mission16/
│
├── data/ # 샘플 데이터 또는 입력 예시
├── models/ # 학습된 모델 및 ONNX 변환 모델
├── report/ # 프로젝트 결과 보고서
├── inference.py # 추론 실행 코드
├── modeling.ipynb # 학습 및 양자화 실험 노트북
├── requirements.txt # 의존성 패키지 목록
└── Summary_Report.pdf # 결과 요약 보고서
```


---

## 4. 결과 요약

- 모델 크기: 약 70% 이상 축소  
- 정밀도 유지율: 98% 수준 유지  
- 추론 속도: ONNX 기반 환경에서 약 1.5~2배 향상  
- PyTorch → ONNX 변환 과정에서 구조적 손실 없음 확인  
- 최적화 결과, 정밀도 대비 연산 효율성 우수

---

## 5. 실행 방법

- 의존성 설치
  ```
  pip install -r requirements.txt
  ```

- 추론 실행
  ```
  python inference.py
  ```
- 결과 보고서 확인
  report/ 또는 Summary_Report.pdf 참고

---

## 6. 버전 관리 및 유지 방침
- ```.venv```, ```__pycache__```, ```.ipynb_checkpoints``` 등은 ```.gitignore```에 의해 버전 관리 제외
- 모델 및 데이터 파일은 최소 단위로 관리
- 대용량 파일은 외부 스토리지 또는 클라우드 저장소 활용 권장
- 커밋 규칙:
  - Add: 신규 파일 추가
  - Fix: 오류 수정
  - Update: 내용 보완
  - Remove: 불필요 파일 삭제
 
  
----------------------

## 작성자
작성자: **AI Engineer Training 3기 김하나**
