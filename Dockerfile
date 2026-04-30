FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN if [ -s requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
COPY . .
ENV PORT=8080
EXPOSE 8080
# Let the application read the PORT env var at runtime (Cloud Run injects it).
# The application defaults to HOST=0.0.0.0 and reads PORT from the environment.
CMD ["python", "recommendation_api.py"]