import os
from dotenv import load_dotenv
from clickhouse_driver import Client
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CH_HOST_ADMIN = os.getenv("CH_HOST", "clickhouse-server")
CH_PORT_ADMIN = int(os.getenv("CH_PORT", 9000))
CH_USER_ADMIN = "default" # Административный пользователь для создания БД
CH_PASSWORD_ADMIN = "your_strong_password" # Пароль для 'default' из users.xml

CH_DB_NAME = os.getenv("CH_DB", "fakenews_db")
CH_USER_BOT = os.getenv("CH_USER", "bot_user")
CH_PASSWORD_BOT = os.getenv("CH_PASSWORD", "bot_password")


def create_database_and_tables():
    try:
        admin_client = Client(
            host=CH_HOST_ADMIN,
            port=CH_PORT_ADMIN,
            user=CH_USER_ADMIN,
            password=CH_PASSWORD_ADMIN
        )
        logging.info(f"Подключение к ClickHouse ({CH_HOST_ADMIN}:{CH_PORT_ADMIN}) от имени {CH_USER_ADMIN}...")

        admin_client.execute(f"CREATE DATABASE IF NOT EXISTS {CH_DB_NAME}")
        logging.info(f"База данных '{CH_DB_NAME}' создана или уже существует.")

        admin_client.disconnect()

        # Подключаемся к созданной БД от имени пользователя бота для создания таблиц
        bot_client = Client(
            host=CH_HOST_ADMIN, # Хост тот же
            port=CH_PORT_ADMIN, # Порт тот же
            user=CH_USER_BOT,
            password=CH_PASSWORD_BOT,
            database=CH_DB_NAME
        )
        logging.info(f"Подключение к БД '{CH_DB_NAME}' от имени {CH_USER_BOT} для создания таблиц...")

        bot_client.execute(f"""
        CREATE TABLE IF NOT EXISTS {CH_DB_NAME}.requests_log (
            request_id UUID DEFAULT generateUUIDv4(),
            user_id Int64,
            chat_id Int64,
            message_id Int64,
            request_timestamp DateTime DEFAULT now(),
            news_text String,
            predicted_label String,
            prediction_probability Float32,
            model_version String DEFAULT '1.0',
            processing_time_ms UInt32
        ) ENGINE = MergeTree()
        ORDER BY (request_timestamp, user_id)
        PARTITION BY toYYYYMM(request_timestamp)
        """)
        logging.info("Таблица 'requests_log' создана или уже существует.")

        bot_client.execute(f"""
        CREATE TABLE IF NOT EXISTS {CH_DB_NAME}.feedback (
            feedback_id UUID DEFAULT generateUUIDv4(),
            request_id UUID,
            user_id Int64,
            feedback_timestamp DateTime DEFAULT now(),
            user_rating String, -- 'correct', 'incorrect'
            user_comment String DEFAULT ''
        ) ENGINE = MergeTree()
        ORDER BY (feedback_timestamp, request_id)
        PARTITION BY toYYYYMM(feedback_timestamp)
        """)
        logging.info("Таблица 'feedback' создана или уже существует.")

        bot_client.disconnect()
        logging.info("Инициализация базы данных завершена успешно.")

    except Exception as e:
        logging.error(f"Ошибка при инициализации базы данных: {e}")
        raise

if __name__ == "__main__":
    import time
    logging.info("Ожидание запуска ClickHouse (15 секунд)...")
    create_database_and_tables()