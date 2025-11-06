# models.py
from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
# 위 줄: SQLModel로 테이블/관계를 정의합니다.

class Movie(SQLModel, table=True):
    # Movie 테이블 정의 (영화 정보 저장)
    id: Optional[int] = Field(default=None, primary_key=True)  # 기본키 (자동증가)
    title: str = Field(index=True)                             # 영화 제목 (검색 용이하도록 인덱스)
    release_date: str                                          # 개봉일 (단순 문자열로 처리)
    director: str                                              # 감독
    genre: str                                                 # 장르
    poster_url: str                                            # 포스터 이미지 URL

    reviews: list["Review"] = Relationship(back_populates="movie")
    # 위 줄: 1(Movie) : N(Review) 관계를 표현 (양방향 연결)

class Review(SQLModel, table=True):
    # Review 테이블 정의 (리뷰 + 감성 결과 저장)
    id: Optional[int] = Field(default=None, primary_key=True)  # 기본키
    movie_id: int = Field(foreign_key="movie.id", index=True)  # 외래키(영화 ID)
    reviewer: str                                              # 작성자명
    content: str                                               # 리뷰 본문
    sentiment: Optional[str] = Field(default=None)             # 감성 결과(예: positive/neutral/negative)
    score: Optional[float] = Field(default=None)               # 감성 점수(예: -1 ~ +1 범위)
    created_at: datetime = Field(default_factory=datetime.utcnow)  # 등록 시각(UTC)

    movie: Optional[Movie] = Relationship(back_populates="reviews")
    # 위 줄: Review -> Movie 역방향 관계 (옵션)
