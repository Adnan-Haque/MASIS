from fastapi import FastAPI
from app.api.document_routes import document_router
from app.api.workspace_routes import workspace_router
from app.db.database import Base, engine
import os
from fastapi.middleware.cors import CORSMiddleware

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # React dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

app.include_router(document_router)
app.include_router(workspace_router)