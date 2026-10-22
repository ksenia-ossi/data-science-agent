FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY ds_agent ./ds_agent
RUN pip install --no-cache-dir .

RUN useradd --create-home agent
USER agent
WORKDIR /work
ENTRYPOINT ["ds-agent"]
