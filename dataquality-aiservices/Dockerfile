# FROM python:3.12-slim

# WORKDIR /

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .

# RUN mkdir -p /root/.streamlit
# COPY .streamlit/secrets.toml /root/.streamlit/secrets.toml

# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# # streamlit will run on port 8501 by default
# EXPOSE 8502

# # run your Streamlit application (important: gui/app.py not cd gui and then run app.py!!!)
# CMD ["streamlit", "run", "streamlit_auto.py", "--server.address=0.0.0.0"]


FROM nvidia/cuda:13.0.1-base-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_15_12.txt .

RUN mkdir -p /root/.streamlit
COPY .streamlit/secrets.toml /root/.streamlit/secrets.toml

RUN pip install --no-cache-dir -r requirements_15_12.txt

COPY . .

EXPOSE 8503

CMD ["streamlit", "run", "streamlit_auto.py", "--server.address=0.0.0.0","--server.port=8501"]
