# API + map only (model outputs are mounted). Training runs on the host (MPS/GPU).
FROM python:3.11-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY greenwatch/__init__.py greenwatch/__init__.py
COPY greenwatch/api greenwatch/api
COPY web web
ENV GW_OUTPUTS=outputs
EXPOSE 8000
CMD ["uvicorn", "greenwatch.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
