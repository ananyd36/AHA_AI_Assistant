FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "cd /app && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 3"]
