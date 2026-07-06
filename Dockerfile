# Serving image for the SHR Offender Profile API.
# Expects trained artifacts in models/ — run `python -m shr.train` first
# (the 318 MB dataset stays out of the image; only the fitted pipelines ship).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shr/ shr/
COPY models/ models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "shr.api:app", "--host", "0.0.0.0", "--port", "8000"]
