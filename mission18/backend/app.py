# app.py
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from pydantic import BaseModel
from backend.db import init_db, get_session
from backend.models import Movie, Review
from backend.sentiment import analyze
# 위 줄들: FastAPI 기본 구성, CORS 설정, DB 세션 주입, 모델/유틸 불러오기

app = FastAPI(title="Movie Review API", description="영화/리뷰/감성분석 API")
# 위 줄: API 문서 제목/설명 (Docs에 표시)

# CORS 설정: Streamlit 프런트엔드에서 접근 가능하도록 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실서비스는 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    # 앱 시작 시 DB 테이블 생성
    init_db()

# ----- Pydantic 요청/응답 스키마 (문서/검증용) -----
class MovieCreate(BaseModel):
    title: str
    release_date: str
    director: str
    genre: str
    poster_url: str

class ReviewCreate(BaseModel):
    movie_id: int
    reviewer: str
    content: str

# ---------------- 영화 관리 ----------------
@app.post("/movies", response_model=Movie)
def create_movie(payload: MovieCreate, session: Session = Depends(get_session)):
    # 영화 등록
    movie = Movie(**payload.model_dump())
    session.add(movie)
    session.commit()
    session.refresh(movie)
    return movie

@app.get("/movies", response_model=List[Movie])
def list_movies(session: Session = Depends(get_session)):
    # 전체 영화 조회
    movies = session.exec(select(Movie)).all()
    return movies

@app.get("/movies/{movie_id}", response_model=Movie)
def get_movie(movie_id: int, session: Session = Depends(get_session)):
    # 특정 영화 조회
    movie = session.get(Movie, movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie

@app.delete("/movies/{movie_id}")
def delete_movie(movie_id: int, session: Session = Depends(get_session)):
    # 특정 영화 삭제
    movie = session.get(Movie, movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    session.delete(movie)
    session.commit()
    return {"ok": True}

# ---------------- 리뷰 관리(+ 작성 시 감성분석) ----------------
@app.post("/reviews", response_model=Review)
def create_review(payload: ReviewCreate, session: Session = Depends(get_session)):
    # 리뷰 등록 + 감성 분석 수행
    movie = session.get(Movie, payload.movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    sentiment, score = analyze(payload.content)
    review = Review(
        movie_id=payload.movie_id,
        reviewer=payload.reviewer,
        content=payload.content,
        sentiment=sentiment,
        score=score,
    )
    session.add(review)
    session.commit()
    session.refresh(review)
    return review

@app.get("/reviews", response_model=List[Review])
def list_reviews(
    movie_id: Optional[int] = None,
    limit: int = 10,
    session: Session = Depends(get_session),
):
    # 전체 또는 특정 영화의 최근 리뷰 조회
    query = select(Review).order_by(Review.created_at.desc())
    if movie_id is not None:
        query = query.where(Review.movie_id == movie_id)
    reviews = session.exec(query).all()
    return reviews[:limit]

@app.delete("/reviews/{review_id}")
def delete_review(review_id: int, session: Session = Depends(get_session)):
    # 특정 리뷰 삭제
    review = session.get(Review, review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    session.delete(review)
    session.commit()
    return {"ok": True}

# ---------------- 평점(감성 점수 평균) ----------------
@app.get("/ratings/{movie_id}")
def get_average_sentiment(movie_id: int, session: Session = Depends(get_session)):
    # 특정 영화의 리뷰 감성 점수 평균
    reviews = session.exec(select(Review).where(Review.movie_id == movie_id)).all()
    if not reviews:
        return {"movie_id": movie_id, "average_score": None, "count": 0}
    valid = [r.score for r in reviews if r.score is not None]
    avg = sum(valid) / len(valid) if valid else None
    return {"movie_id": movie_id, "average_score": avg, "count": len(valid)}

# ---------------- (옵션) 단일 문장 감성 분석 ----------------
@app.get("/sentiment/analyze")
def analyze_text(text: str):
    # 단일 텍스트 감성 분석 (테스트/디버깅용)
    label, score = analyze(text)
    return {"label": label, "score": score}
