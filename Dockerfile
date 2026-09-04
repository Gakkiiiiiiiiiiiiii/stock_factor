FROM python:3.12-slim AS base
WORKDIR /app
ENV STOCK_FACTOR_CONFIG_ROOT=/app/config
COPY pyproject.toml ./
COPY src ./src
COPY contracts ./contracts
COPY config ./config
COPY locks ./locks

# The core image deliberately installs no torch/GPU packages.
FROM base AS core
RUN pip install --no-cache-dir --no-deps . \
    && pip install --no-cache-dir --no-deps -r locks/core.lock \
    && pip check
LABEL org.stock-factor.profile="core"
EXPOSE 8200
CMD ["python", "-m", "uvicorn", "stock_factor.api.main:app", "--host", "0.0.0.0", "--port", "8200"]

FROM core AS worker
LABEL org.stock-factor.profile="worker"
CMD ["python", "-m", "stock_factor.workers.factor_worker"]

FROM base AS ml-cpu
ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --no-deps . \
    && grep -v -E '^torch==' locks/ml-cpu.lock > /tmp/ml-cpu.lock \
    && pip install --no-cache-dir --no-deps -r /tmp/ml-cpu.lock \
    && pip install --no-cache-dir --no-deps --index-url "${TORCH_CPU_INDEX_URL}" torch==2.11.0+cpu \
    && pip check
LABEL org.stock-factor.profile="ml-cpu"
CMD ["python", "-m", "uvicorn", "stock_factor.api.main:app", "--host", "0.0.0.0", "--port", "8200"]

# GPU builds are explicitly tied to the CUDA wheel index.  The CUDA base image
# and runner are selected by the deployment environment, not imported by core.
FROM ml-cpu AS ml-gpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir --no-deps . \
    && pip uninstall --yes torch \
    && grep -v -E '^torch==' locks/ml-gpu.lock > /tmp/ml-gpu.lock \
    && pip install --no-cache-dir --no-deps -r /tmp/ml-gpu.lock \
    && pip install --no-cache-dir --no-deps --index-url "${TORCH_INDEX_URL}" torch==2.11.0+cu128 \
    && pip check
LABEL org.stock-factor.profile="ml-gpu" org.stock-factor.torch-index="https://download.pytorch.org/whl/cu128"
CMD ["python", "-m", "uvicorn", "stock_factor.api.main:app", "--host", "0.0.0.0", "--port", "8200"]
