FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# REMOVE these two lines for now:
# RUN mkdir -p /root/.streamlit
# COPY .streamlit/secrets.toml /root/.streamlit/secrets.toml

COPY . .
#COPY app.py ./app.py

EXPOSE 8502
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8502"]
