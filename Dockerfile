# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /code

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        tesseract-ocr \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the app folder
COPY ./app ./app

# Set Python path
ENV PYTHONPATH=/code

# Default command (overridden by docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]