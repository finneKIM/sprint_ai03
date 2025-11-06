# app.py
import streamlit as st                 # 스트림릿 UI 프레임워크
import requests                        # 백엔드(FastAPI)와 통신
from datetime import datetime          # 날짜 표시용

BACKEND = st.secrets.get("BACKEND_URL", "http://127.0.0.1:8000")
# 위 줄: 배포 시 Streamlit Secrets로 BACKEND_URL 설정 가능. 로컬은 기본 8000 포트 사용.

st.set_page_config(page_title="🎬 영화 리뷰 감성 분석", layout="wide")
# 위 줄: 페이지 제목/레이아웃 설정

st.title("🎬 영화 리뷰 감성 분석 웹앱")
# 위 줄: 상단 타이틀

# --------------- 섹션: 영화 등록 ---------------
st.header("➕ 영화 추가")
with st.form("movie_form", clear_on_submit=True):
    title = st.text_input("제목")
    release_date = st.text_input("개봉일 (예: 2020-01-01)")
    director = st.text_input("감독")
    genre = st.text_input("장르")
    poster_url = st.text_input("포스터 URL")
    submitted = st.form_submit_button("영화 등록")
    # 위 줄: 사용자가 입력하고 버튼 클릭 시 제출

    if submitted:
        if not title or not release_date or not director or not genre or not poster_url:
            st.error("모든 필드를 입력해주세요.")
        else:
            payload = {
                "title": title,
                "release_date": release_date,
                "director": director,
                "genre": genre,
                "poster_url": poster_url,
            }
            resp = requests.post(f"{BACKEND}/movies", json=payload)
            if resp.status_code == 200:
                st.success("영화가 등록되었습니다!")
            else:
                st.error(f"등록 실패: {resp.text}")

st.divider()

# --------------- 섹션: 영화 목록 ---------------
st.header("🎞️ 영화 목록")
movies = requests.get(f"{BACKEND}/movies").json()
cols = st.columns(3)
for i, m in enumerate(movies):
    with cols[i % 3]:
        st.subheader(f"[{m['id']}] {m['title']}")
        if m.get("poster_url"):
            st.image(m["poster_url"], use_container_width=True)
        st.caption(f"{m['release_date']} · {m['director']} · {m['genre']}")

st.divider()

# --------------- 섹션: 리뷰 작성 ---------------
st.header("🗣️ 리뷰 작성")
if not movies:
    st.info("먼저 영화를 하나 이상 등록해주세요.")
else:
    movie_options = {f"[{m['id']}] {m['title']}": m["id"] for m in movies}
    selected = st.selectbox("영화 선택", list(movie_options.keys()))
    movie_id = movie_options[selected]
    reviewer = st.text_input("작성자 이름")
    content = st.text_area("리뷰 내용", height=120)

    if st.button("리뷰 등록(+ 감성 분석)"):
        if not reviewer or not content:
            st.error("작성자와 리뷰 내용을 입력해주세요.")
        else:
            payload = {"movie_id": movie_id, "reviewer": reviewer, "content": content}
            r = requests.post(f"{BACKEND}/reviews", json=payload)
            if r.status_code == 200:
                data = r.json()
                st.success(f"등록 완료! 감성: {data.get('sentiment')} (score={data.get('score'):.3f})")
            else:
                st.error(f"등록 실패: {r.text}")

st.divider()

# --------------- 섹션: 최근 리뷰(10개) ---------------
st.header("📋 최근 리뷰 (최대 10개)")
reviews = requests.get(f"{BACKEND}/reviews", params={"limit": 10}).json()
if not reviews:
    st.info("등록된 리뷰가 아직 없습니다.")
else:
    for rv in reviews:
        with st.container(border=True):
            st.markdown(f"**영화ID:** {rv['movie_id']}  |  **작성자:** {rv['reviewer']}  |  **감성:** `{rv.get('sentiment')}`  |  **점수:** `{rv.get('score')}`")
            ts = rv.get("created_at", None)
            st.caption(f"등록일: {ts}")
            st.write(rv["content"])

st.divider()

# --------------- 섹션: 영화별 평균 감성 점수 ---------------
st.header("⭐ 영화별 평균 감성 점수")
if movies:
    col_a, col_b = st.columns(2)
    with col_a:
        pick = st.selectbox("평균 점수 조회할 영화 선택", list(movie_options.keys()), key="avg1")
        movie_id_avg = movie_options[pick]
        avg = requests.get(f"{BACKEND}/ratings/{movie_id_avg}").json()
        st.write(f"**평균 점수:** {avg['average_score']}  (count={avg['count']})")
    with col_b:
        st.info("점수 범위는 대략 -1(부정) ~ +1(긍정). 0에 가까울수록 중립입니다.")
