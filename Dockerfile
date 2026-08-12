FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
EXPOSE 8200
CMD ["python", "-m", "uvicorn", "stock_factor.api.main:app", "--host", "0.0.0.0", "--port", "8200"]
