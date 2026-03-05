FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Debug: confirm frontend is present at build time
RUN echo "=== Build context check ===" && \
    ls -la /app/ && \
    ls -la /app/frontend/

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "cd /app/backend && python main.py"]
