FROM python:3.11-slim

WORKDIR /app

#System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ython dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy project
COPY . .

#Pre-run training on container start
RUN python entrypoint/train.py

#Default: run inference demo
CMD ["python", "entrypoint/inference.py", "--all"]
