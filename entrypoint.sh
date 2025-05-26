#!/bin/bash
set -e

CH_HOST_TO_CHECK="${CH_HOST:-clickhouse-server}"
CH_PORT_TO_CHECK="${CH_PORT:-9000}"

echo "Entrypoint: Waiting for ClickHouse at $CH_HOST_TO_CHECK:$CH_PORT_TO_CHECK..."


timeout_seconds=120
start_time=$(date +%s)

while ! nc -z $CH_HOST_TO_CHECK $CH_PORT_TO_CHECK; do
  current_time=$(date +%s)
  elapsed_time=$((current_time - start_time))

  if [ $elapsed_time -ge $timeout_seconds ]; then
    echo "Entrypoint: Timeout waiting for ClickHouse. Exiting."
    exit 1
  fi
  echo "Entrypoint: ClickHouse is not yet available. Retrying in 5 seconds..."
  sleep 5
done

echo "Entrypoint: ClickHouse is up!"


echo "Entrypoint: Running database migrations..."
python init_db.py

echo "Entrypoint: Starting Telegram bot..."
exec python -m bot.main