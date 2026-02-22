from celery import Celery

celery_app = Celery(
    "masis",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0"
)


# IMPORTANT: import tasks so they register
import app.workers.ingestion_tasks