from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://masis:masis@postgres:5432/masis"

engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_timeout=30,
    pool_recycle=1800,
)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()