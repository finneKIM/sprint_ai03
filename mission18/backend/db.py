# db.py
from sqlmodel import SQLModel, create_engine, Session  # SQLModel: SQLAlchemy + Pydantic 통합 레이어
# 위 줄: SQLModel은 테이블/스키마를 손쉽게 정의하고 DB연결을 도와줍니다.

DATABASE_URL = "sqlite:///./movies.db"  # 현재 폴더에 movies.db 파일을 생성/사용
# 위 줄: SQLite 파일 DB 경로 설정 (로컬 개발과 과제에 적합)

engine = create_engine(DATABASE_URL, echo=False)  # 엔진 생성 (echo=True면 SQL 로그 출력)
# 위 줄: DB와의 실제 연결 객체. 애플리케이션 전체에서 재사용합니다.

def init_db() -> None:
    # DB 초기화(테이블 생성). 이미 있으면 그대로 둡니다.
    SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    # 요청마다 DB 세션을 하나 열고, 요청 종료 시 닫기 위해 사용합니다(의존성).
    with Session(engine) as session:
        yield session
