# sentiment.py
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# # 위 줄: VADER는 가벼운 룰 기반 감성 분석기(영문 중심). 데모/실습에 적합(빠름, 의존성 작음).

# _analyzer = SentimentIntensityAnalyzer()
# # 위 줄: 애플리케이션 구동 시 한 번만 로드하여 재사용 (경량 모델처럼 사용)

# def analyze(text: str) -> tuple[str, float]:
#     """
#     입력 텍스트를 분석해 (라벨, 점수)를 반환.
#     - 라벨: 'positive' | 'neutral' | 'negative'
#     - 점수: compound 점수(-1.0 ~ +1.0)
#     """
#     if not text or not text.strip():
#         return "neutral", 0.0  # 빈 문장은 중립 처리
#     scores = _analyzer.polarity_scores(text)
#     compound = scores.get("compound", 0.0)
#     if compound >= 0.05:
#         label = "positive"
#     elif compound <= -0.05:
#         label = "negative"
#     else:
#         label = "neutral"
#     return label, compound



# ================================ 성능 별로여서 교체
# backend/sentiment.py
# Transformer 기반 다국어 감성분석 (3-class: negative/neutral/positive)
# - 모델: cardiffnlp/twitter-xlm-roberta-base-sentiment
# - 반환(label, score): label은 {'negative','neutral','positive'}
#   score는 [-1.0, +1.0] 범위로 변환( pos_prob - neg_prob )


from typing import Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, use_fast=False)
_model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)

# 동적 양자화: Linear 레이어를 INT8로 (CPU 가속)
_model = torch.quantization.quantize_dynamic(
    _model, {torch.nn.Linear}, dtype=torch.qint8
)

_model.eval()

def analyze(text: str) -> Tuple[str, float]:
    if not text or not text.strip():
        return "neutral", 0.0

    inputs = _tokenizer(text, max_length=256, truncation=True, padding=False, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits  # [1, 3]
        probs = F.softmax(logits, dim=-1).squeeze(0)  # [neg, neu, pos]

    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label_id = int(torch.argmax(probs).item())
    label = id2label[label_id]

    score = float(probs[2].item() - probs[0].item())  # pos - neg ∈ [-1, 1]
    return label, score
















# # =================================onnx로 양자화 실패(나중에 다시 해보기)
# from typing import Tuple
# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer
# from optimum.onnxruntime import ORTModelForSequenceClassification

# # 1) 모델과 토크나이저를 앱 시작 시 1회 로드(메모리 캐시)
# _MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, use_fast=False)
# # ORT 모델: 없으면 최초 1회 export 후 캐시 사용(처음 1회만 시간이 조금 걸릴 수 있음)
# _model = ORTModelForSequenceClassification.from_pretrained(
#     _MODEL_NAME,
#     export=True,
#     from_transformers=True
# )

# def _softmax(x: np.ndarray) -> np.ndarray:
#     e = np.exp(x - np.max(x, axis=-1, keepdims=True))
#     return e / e.sum(axis=-1, keepdims=True)

# def analyze(text: str) -> Tuple[str, float]:
#     if not text or not text.strip():
#         return "neutral", 0.0

#     # 바로 NumPy로 받기 → 변환 비용, torch 의존 줄임
#     inputs = _tokenizer(
#         text,
#         max_length=256,
#         truncation=True,
#         padding=False,
#         return_tensors="np"
#     )

#     # ORT는 numpy 입력을 기본으로 지원
#     outputs = _model(**inputs)
#     logits = outputs.logits  # shape: (1, 3) numpy array
#     probs = _softmax(logits)[0]  # [neg, neu, pos]

#     id2label = {0: "negative", 1: "neutral", 2: "positive"}
#     label_id = int(probs.argmax())
#     label = id2label[label_id]

#     score = float(probs[2] - probs[0])  # pos - neg ∈ [-1, 1]
#     return label, score