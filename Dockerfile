FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_tg.txt .
RUN pip install -r requirements_tg.txt

RUN python -m nltk.downloader wordnet stopwords punkt omw-1.4 punkt_tab

COPY . .

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]