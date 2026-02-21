
#Its a dockerfile
FROM python:3.11-slim


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
#RUN chown -R appuser:appgroup /app      
# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*
# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY gunicorn.conf.py .
COPY src/ /app/src/
#COPY artifacts /app/artifacts
#COPY output /app/output

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
 CMD wget -qO- http://127.0.0.1:8000/health || exit 1
# Run as non-root
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

# IMPORTANT: Linux entrypoint via gunicorn
CMD ["gunicorn", "--config", "gunicorn.conf.py", "aiservices.wsgi:app"]
