import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import onnxruntime as ort
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 구조 재정의
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),   # ← Lienar 오타 수정
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)      # 합성곱 특징 추출 경로 적용
        x = self.classifier(x)    # 분류기 경로 적용
        return x                  # 로짓 반환




def get_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

test_loader = get_test_loader()


# ------- 공통 유틸 함수 -------
def file_mb(path):
    """파일 용량을 MB 단위로 반환"""
    import os
    return os.path.getsize(path) / (1024 * 1024)



# ------- PyTorch FP32 --------
fp32_path = "models/mission_16_mnist_cnn.pth"  # ← 슬래시 사용 (윈/맥 공통)
assert os.path.exists(fp32_path), "FP32 가중치 파일 누락"

model_fp32 = MnistCNN().to(device)
state = torch.load(fp32_path, map_location=device)
model_fp32.load_state_dict(state)
model_fp32.eval()

def eval_torch(model):
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

acc_fp32 = eval_torch(model_fp32)
print(f"FP32 .pth Acc: {acc_fp32:.4f}")


# ------- PyTorch INT8 (동적 양자화 저장본) --------
int8_path = "models/mission_16_mnist_cnn_int8.pth"
assert os.path.exists(int8_path), "INT8 가중치 파일 누락"

# 1) FP32 구조를 만든 뒤,
model_int8 = MnistCNN()  # <= 여기서는 .to(device) 하지 않음 (양자화=CPU 전용)

# 2) 해당 구조에 '동적 양자화'를 동일하게 적용
model_int8 = torch.quantization.quantize_dynamic(
    model_int8, {nn.Linear}, dtype=torch.qint8
)

# 3) 양자화 state_dict 로드 (CPU에 로드)
state_int8 = torch.load(int8_path, map_location="cpu")
model_int8.load_state_dict(state_int8)
model_int8.eval()

# 4) INT8 평가는 CPU에서 (입력도 CPU 텐서)
def eval_torch_cpu(model):
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            # x, y는 기본이 CPU 텐서 → 그대로 사용
            logits = model(x)                 # 동적 양자화 Linear는 CPU 경로
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

acc_int8 = eval_torch_cpu(model_int8)
print(f"INT8 .pth Acc: {acc_int8:.4f}")



# ------- ONNX (onnxruntime) --------
onnx_path = "models/mission_16_mnist_cnn.onnx"   # ← 폴더/파일명 오타 수정
assert os.path.exists(onnx_path), "ONNX 파일 누락"

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

def eval_onnx(session):
    correct, total = 0, 0
    for x, y in test_loader:
        x_np = x.numpy().astype(np.float32)
        out = session.run(None, {input_name: x_np})
        logits = out[0]
        pred = logits.argmax(axis=1)
        correct += (pred == y.numpy()).sum()
        total += y.size(0)
    return correct / total

acc_onnx = eval_onnx(session)
print(f"ONNX Acc: {acc_onnx:.4f}")



# ------- 간단 속도 비교 --------
def bench_torch(model, iters=50):
    model.eval()
    x, _ = next(iter(test_loader))
    x = x.to(device)
    with torch.no_grad():
        model(x)  # warmup
        t0 = time.time()
        for _ in range(iters):
            model(x)
        dt = (time.time() - t0) / iters
    return dt

def bench_onnx(session, iters=50):
    x, _ = next(iter(test_loader))
    x_np = x.numpy().astype(np.float32)
    session.run(None, {input_name: x_np})  # warmup
    t0 = time.time()
    for _ in range(iters):
        session.run(None, {input_name: x_np})
    dt = (time.time() - t0) / iters
    return dt

t_fp32 = bench_torch(model_fp32)
t_int8 = bench_torch(model_int8)
t_onnx = bench_onnx(session)

print("Avg inference time per batch (approx):")
print(f"- FP32 PyTorch : {t_fp32*1000:.2f} ms")
print(f"- INT8  PyTorch: {t_int8*1000:.2f} ms")
print(f"- ONNX Runtime : {t_onnx*1000:.2f} ms")



os.makedirs("report", exist_ok=True)
# ------- 결과 요약 저장 --------
os.makedirs("report", exist_ok=True)  # report 폴더가 없으면 자동 생성
with open("report/inference_summary.txt", "w", encoding="utf-8") as f:
    f.write(
        f"FP32 Acc: {acc_fp32:.4f}\n"
        f"INT8 Acc: {acc_int8:.4f}\n"
        f"ONNX Acc: {acc_onnx:.4f}\n"
        f"FP32 time(ms): {t_fp32*1000:.2f}\n"
        f"INT8 time(ms): {t_int8*1000:.2f}\n"
        f"ONNX time(ms): {t_onnx*1000:.2f}\n"
    )


# ------- 모델 용량 그래프 (겹침 방지/가독성 강화 버전) -------
import os
import matplotlib.pyplot as plt

def file_mb(path):
    return os.path.getsize(path) / (1024*1024)

fp32_path = os.path.join("models", "mission_16_mnist_cnn.pth")
int8_path = os.path.join("models", "mission_16_mnist_cnn_int8.pth")
onnx_path = os.path.join("models", "mission_16_mnist_cnn.onnx")
for p in [fp32_path, int8_path, onnx_path]:
    assert os.path.exists(p), f"누락: {p}"

sizes = {"FP32 .pth": file_mb(fp32_path),
         "INT8 .pth": file_mb(int8_path),
         "ONNX":      file_mb(onnx_path)}

labels = list(sizes.keys())
values = [sizes[k] for k in labels]

fig, ax = plt.subplots(figsize=(7.0, 6.0))          # 세로 공간 여유
bars = ax.bar(labels, values)

# 막대 위 용량 라벨(막대 높이에 비례해 간격 확보)
offset = max(values) * 0.005
for i, v in enumerate(values):
    ax.text(i, v + offset, f"{v:.2f} MB", ha="center", va="bottom")

ax.set_ylabel("Size (MB)")
ax.set_title("Mission16 Model Size Comparison")

# x축 눈금과 그래프 하단 사이 간격 확보
ax.tick_params(axis='x')

# 하단 여백 크게 확보 (정확도 텍스트 들어갈 자리)
plt.subplots_adjust(bottom=0.20)

# 정확도 텍스트를 3개로 분리해서 하단에 박스로 배치 (겹침 방지)
box = dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.25')
fig.text(0.18, 0.06, f"FP32: {acc_fp32:.4f}", ha='center', va='center', bbox=box)
fig.text(0.50, 0.06, f"INT8: {acc_int8:.4f}", ha='center', va='center', bbox=box)
fig.text(0.82, 0.06, f"ONNX: {acc_onnx:.4f}", ha='center', va='center', bbox=box)
# fig.text(0.5, 0.095, "Accuracy", ha='center')

# 저장
os.makedirs("report", exist_ok=True)
out_path = os.path.join("report", "sizes_and_results.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"[OK] 저장 완료: {out_path}")


# csv로 저장
import csv
os.makedirs("report", exist_ok=True)
with open("report/inference_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["format","accuracy","time_ms","size_mb"])
    w.writerow(["FP32 .pth", acc_fp32, t_fp32*1000, file_mb("models/mission_16_mnist_cnn.pth")])
    w.writerow(["INT8 .pth", acc_int8, t_int8*1000, file_mb("models/mission_16_mnist_cnn_int8.pth")])
    w.writerow(["ONNX",      acc_onnx, t_onnx*1000, file_mb("models/mission_16_mnist_cnn.onnx")])
