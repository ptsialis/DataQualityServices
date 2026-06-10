FROM continuumio/miniconda3:24.3.0-0

RUN conda install -y python=3.11 \
    && conda clean -afy

ENV PATH=/opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install shared and backend-specific Python dependencies
COPY requirements-base.txt /tmp/requirements-base.txt
COPY BE/requirements.txt /tmp/requirements-backend.txt

RUN python -m pip install \
        --upgrade \
        "pip<25" \
        "setuptools<70" \
        wheel \
    && python -m pip install \
        --no-cache-dir \
        -r /tmp/requirements-base.txt \
        -r /tmp/requirements-backend.txt

# Download required NLTK resources
ENV NLTK_DATA=/usr/local/share/nltk_data

RUN mkdir -p /usr/local/share/nltk_data \
    && python - <<'PY'
import nltk

download_dir = "/usr/local/share/nltk_data"

for package in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(package, download_dir=download_dir)
PY

RUN mkdir -p /usr/local/share/nltk_data/tokenizers/punkt/PY3 \
    && cp /usr/local/share/nltk_data/tokenizers/punkt/*.pickle \
          /usr/local/share/nltk_data/tokenizers/punkt/PY3/ \
       || true

ENV DGLBACKEND=pytorch
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
ENV FLASK_ENV=development

COPY BE/. /app

EXPOSE 5000

CMD ["python", "app.py"]
