FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/.streamlit
COPY .streamlit/secrets.toml /root/.streamlit/secrets.toml

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# COPY gui/ ./gui
# COPY service/ ./service/
# COPY assets/ ./assets/
# COPY docs/ ./docs/
# COPY datasets/ ./datasets/
# COPY model_weights/ ./model_weights/
# COPY main.py ./main.py
# COPY __init__.py ./__init__.py


EXPOSE 8501

CMD ["streamlit", "run", "gui/app.py", "--server.address=0.0.0.0","--server.port=8501"]
