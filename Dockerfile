FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source
COPY . .

# Railway injects $PORT at runtime
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "cd backend && python main.py"]
