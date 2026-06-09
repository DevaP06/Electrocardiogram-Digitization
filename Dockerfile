FROM python:3.12-slim

WORKDIR /app

# Install PyTorch CPU-only wheels first (separate layer for caching)
RUN pip install --no-cache-dir \
    torch==2.12.0 \
    torchvision==0.27.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY src/ src/
COPY weights/ weights/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
