# ui.py
import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 로직 함수들은 app.py에서 import
from app import load_model, preprocess_from_canvas, predict, ensure_model_file

st.set_page_config(page_title="MNIST 손글씨 인식 데모", layout="wide")
st.title("MNIST 손글씨 숫자 인식 (ONNX + Streamlit)")
st.caption("그림을 그리면 0~9 확률을 예측합니다. 배경은 흰색, 펜은 검정색 권장.")

# -----------------------------
# 세션 상태 초기값
# -----------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []          # 예측 히스토리 저장소
if "last_chart_df" not in st.session_state:
    st.session_state["last_chart_df"] = None  # 마지막 예측 확률 막대 그래프용 DF

# 모델 파일 보장(경고는 UI로)
_ = ensure_model_file(on_warning=st.warning)

@st.cache_resource(show_spinner=True)
def get_session():
    return load_model()

# =========================================================
# 레이아웃
#   [좌 2] (캔버스 | 전처리 미리보기)   |   [우 1] 히스토리
#                         ↓
#                     [아래] 예측 차트
# =========================================================

# 좌-우 큰 컬럼: 좌측을 넓게(2), 우측을 좁게(1)
left_wide, right_narrow = st.columns([2, 1], gap="large")

with left_wide:
    # 좌측을 다시 두 컬럼으로 쪼갬: (1) 입력 캔버스 | (2) 전처리 이미지
    col_canvas, col_preview = st.columns([1, 1], gap="medium")

    with col_canvas:
        st.subheader("1) 입력 캔버스")
        canvas_result = st_canvas(
            fill_color="#00000000",            # 투명 채우기
            # stroke_width=20,         # 펜 굵기
            stroke_color="#000000",            # 검정 펜
            background_color="#FFFFFF",        # 흰 배경
            width=280, height=280,             # 그리는 영역
            drawing_mode="freedraw",
            key="canvas",
        )
        stroke_width = st.slider("펜 두께", 10, 100, 20)  # 기본 20으로 보기 좋게

    with col_preview:
        st.subheader("2) 전처리 이미지(28×28, 반전/정규화)")
        preview_img = None                     # 전처리 미리보기 이미지 보관 변수
        input_tensor = None                    # 모델 입력 텐서 보관 변수

        if canvas_result.image_data is not None:
            # 캔버스 numpy RGBA → PIL.Image
            canvas_img = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA")
            # 전처리: (1,1,28,28) + 28x28 시각화 이미지
            input_tensor, preview_img = preprocess_from_canvas(canvas_img)

            # 전처리 결과 썸네일 표시 (Pillow 10: Resampling.NEAREST 권장)
            st.image(
                preview_img.resize((280, 280), Image.Resampling.NEAREST),
                caption="모델 입력 미리보기(확대)",
                width=280
            )

            # 예측 버튼 (전처리 패널에 둠)
            session = get_session()
            if session is None:
                st.error("모델 로드 실패: models/mnist.onnx를 수동 배치 후 다시 실행하세요.")
            else:
                if st.button("예측하기", type="primary"):
                    result = predict(session, input_tensor)
                    probs = result["probs"]
                    label = result["label"]

                    st.success(f"예측 라벨: **{label}**")

                    # 차트용 DF는 세션 상태에 저장 → 아래 ‘예측 차트’ 영역에서 출력
                    chart_df = pd.DataFrame({"digit": list(range(10)), "probability": probs}).set_index("digit")
                    st.session_state["last_chart_df"] = chart_df

                    # 히스토리(오른쪽 패널에서 보여줄 데이터) 추가
                    top3_idx = np.argsort(-probs)[:3].tolist()
                    top3 = [(int(i), float(probs[i])) for i in top3_idx]

                    # 원본/전처리 썸네일을 바이트로 저장(세션 직렬화 안전)
                    thumb_orig = canvas_img.convert("RGB").resize((84, 84))
                    buf_orig = io.BytesIO()
                    thumb_orig.save(buf_orig, format="PNG")
                    buf_orig.seek(0)

                    buf_proc = io.BytesIO()
                    preview_img.resize((84, 84), Image.Resampling.NEAREST).save(buf_proc, format="PNG")
                    buf_proc.seek(0)

                    st.session_state["history"].append({
                        "label": label,
                        "top3": top3,
                        "orig_png": buf_orig.getvalue(),
                        "proc_png": buf_proc.getvalue(),
                    })

with right_narrow:
    st.subheader("3) 이미지 저장소(히스토리)")
    if len(st.session_state["history"]) == 0:
        st.info("아직 저장된 기록이 없습니다. 왼쪽에서 그림을 그리고 '예측하기'를 눌러보세요.")
    else:
        for item in reversed(st.session_state["history"]):
            c1, c2, c3 = st.columns([1, 1, 1.6])
            with c1:
                st.caption("원본")
                st.image(item["orig_png"])
            with c2:
                st.caption("전처리")
                st.image(item["proc_png"])
            with c3:
                st.caption("결과")
                st.write(f"라벨: **{item['label']}**")
                top3_txt = ", ".join([f"{d}:{p:.2f}" for d, p in item["top3"]])
                st.write(f"Top3: {top3_txt}")

        st.markdown("---")
        if st.button("히스토리 비우기"):
            st.session_state["history"].clear()
            st.session_state["last_chart_df"] = None
            st.success("히스토리를 삭제했습니다.")

# -----------------------------
# (4) 예측 차트: 페이지 하단 전용 영역
# -----------------------------
st.markdown("### 4) 예측 확률 차트")
if st.session_state["last_chart_df"] is not None:
    st.bar_chart(st.session_state["last_chart_df"])
else:
    st.caption("아직 예측 결과가 없습니다. 좌측에서 그림을 그리고 ‘예측하기’를 눌러보세요.")
