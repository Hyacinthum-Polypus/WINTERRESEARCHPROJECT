FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Optional build tools if you ever need to compile wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps via pip (not apt)
RUN pip install --no-cache-dir pymupdf python-arango pymupdf4llm pathlib