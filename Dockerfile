FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Flashrank model so the first reranking request isn't slow
RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2')"

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "python -c 'import main' && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
