FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    "aiohttp>=3.9.0" \
    "orjson>=3.9.0" \
    "argostranslate>=1.9.0" \
    "nltk>=3.8.0" \
    "huggingface_hub==0.26.5" \
    "transformers==4.46.3" \
    "sentence-transformers==3.3.1" \
    "vaderSentiment>=3.3.2" \
    "finvader>=1.0.0" \
    httpx[socks] socksio \
    "aioprometheus>=23.0.0" \
    yake \
    "tiktoken>=0.5.0"

# exorde_data без рекурсивных сабмодулей — см. комментарий в bpipe/Dockerfile
RUN git clone --branch full --single-branch --no-recurse-submodules --depth 1 \
    https://github.com/exorde-labs/exorde_data.git /tmp/exorde_data \
    && pip install --no-cache-dir /tmp/exorde_data \
    && rm -rf /tmp/exorde_data

RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

COPY . .

CMD ["python", "upipe.py"]
