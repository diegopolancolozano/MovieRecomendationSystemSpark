FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN if [ -s requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
COPY . .
ENV PORT=8080
EXPOSE 8080
CMD ["python", "recommendation_api.py", "--host", "0.0.0.0", "--port", "8080"]