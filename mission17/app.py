# core.py
import os
import numpy as np
import requests
from PIL import Image, ImageOps
import onnxruntime as ort

# === 모델 자동 다운로드 유틸 ===
import os
import requests

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mnist.onnx")
MNIST_ONNX_URL = (
    "https://github.com/onnx/models/raw/main/"
    "validated/vision/classification/mnist/model/mnist-8.onnx"
)

def ensure_model_file(on_warning=None) -> str:
    """
    1) models/ 디렉토리 생성
    2) models/mnist.onnx 없으면 GitHub에서 다운로드
    3) 성공하면 파일 경로 반환, 실패하면 "" 반환
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    try:
        # 스트리밍 다운로드(대용량 안전)
        with requests.get(MNIST_ONNX_URL, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        # 간단한 무결성 확인(파일 크기 확인 정도)
        if os.path.getsize(MODEL_PATH) < 10_000:  # 10KB 미만이면 비정상
            raise RuntimeError("Downloaded file size too small.")
        return MODEL_PATH

    except Exception as e:
        if on_warning:
            on_warning(
                "모델 자동 다운로드에 실패했습니다. 방화벽/네트워크를 확인하거나 "
                "아래 링크에서 `mnist-8.onnx`를 직접 내려받아 "
                f"`{MODEL_PATH}` 위치에 파일명을 `mnist.onnx`로 저장하세요.\n\n"
                "링크: https://github.com/onnx/models/tree/main/validated/vision/classification/mnist/model"
            )
        return ""


def load_model() -> ort.InferenceSession | None:
    """
    ONNX Runtime InferenceSession 생성 (캐싱은 UI쪽에서 담당)
    """
    model_path = ensure_model_file()
    if not model_path or not os.path.exists(model_path):
        return None

    sess_opts = ort.SessionOptions()
    providers = ort.get_available_providers()
    session = ort.InferenceSession(model_path, sess_opts, providers=providers)
    return session

def preprocess_from_canvas(canvas_img: Image.Image) -> tuple[np.ndarray, Image.Image]:
    """
    캔버스 PIL 이미지를 MNIST 입력(1,1,28,28 float32, 배경0/글자1)로 변환
    그리고 전처리된 28x28 이미지를 함께 반환(시각화용)
    """
    if canvas_img.mode != "RGBA":
        canvas_img = canvas_img.convert("RGBA")

    white_bg = Image.new("RGBA", canvas_img.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(white_bg, canvas_img)

    gray = composite.convert("L")
    gray_28 = gray.resize((28, 28), Image.BILINEAR)
    gray_28_inv = ImageOps.invert(gray_28)            # 배경=0, 글자=1 기대치 맞춤

    arr = np.array(gray_28_inv, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)   # (1, 28, 28)
    arr = np.expand_dims(arr, axis=0)   # (1, 1, 28, 28)

    return arr, gray_28_inv


def softmax(x: np.ndarray) -> np.ndarray:
    # 오버플로 방지: 최대값을 빼고 exp
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def predict(session, input_tensor):
    input_name = "Input3"
    output_name = "Plus214_Output_0"

    outputs = session.run([output_name], {input_name: input_tensor})
    logits = outputs[0]                      # (1, 10)

    # 소프트맥스 직접 계산
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = e / np.sum(e, axis=-1, keepdims=True)
    probs = probs[0]                         # (10,)

    label = int(np.argmax(probs))
    return {"probs": probs, "label": label}

