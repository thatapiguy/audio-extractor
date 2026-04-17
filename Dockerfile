FROM python:3.11-slim

# Install ffmpeg (provides both ffmpeg and ffprobe)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY app.py .

# Streamlit config — disable the default browser-open behaviour and telemetry
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Persist YOTO credentials across container restarts by mounting a host dir here:
#   -v /share/containers/audio-extractor/data:/root
# (the app writes ~/.yoto_credentials.json)
VOLUME ["/root"]

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
