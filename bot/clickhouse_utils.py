import logging
import uuid
from clickhouse_driver import Client
from .config import CH_HOST, CH_PORT, CH_USER, CH_PASSWORD, CH_DB, MODELS_CONFIG


db_client_instance = None

def get_clickhouse_client():
    global db_client_instance
    reconnect_needed = False

    if db_client_instance is None:
        reconnect_needed = True
    else:
        try:
            db_client_instance.execute("SELECT 1")
        except Exception as e:
            logging.warning(f"Соединение с ClickHouse потеряно или недействительно: {e}. Попытка переподключения.")
            reconnect_needed = True
            if db_client_instance and getattr(db_client_instance, 'connection', None): # Проверка атрибута connection
                try:
                    db_client_instance.disconnect()
                except Exception:
                    pass
            db_client_instance = None

    if reconnect_needed:
        try:
            logging.info(f"Попытка подключения к ClickHouse ({CH_HOST}:{CH_PORT}, БД: {CH_DB})...")
            db_client_instance = Client(
                host=CH_HOST, port=CH_PORT, user=CH_USER, password=CH_PASSWORD, database=CH_DB,
                connect_timeout=10, send_receive_timeout=300,
                settings={'max_block_size': 100000}
            )
            db_client_instance.execute("SELECT 1")
            logging.info(f"Успешное новое подключение к ClickHouse ({CH_HOST}:{CH_PORT}, БД: {CH_DB})")
        except Exception as e:
            logging.error(f"Ошибка нового подключения к ClickHouse: {e}", exc_info=True)
            db_client_instance = None
    return db_client_instance

async def log_request_to_db(user_id: int, chat_id: int, message_id: int,
                            news_text: str, predicted_label: str,
                            probability: float | None, processing_time_ms: int,
                            selected_model_id: str) -> str | None:
    client = get_clickhouse_client()
    if not client:
        logging.warning("Нет подключения к БД, логирование запроса пропускается.")
        return None

    request_id_uuid = uuid.uuid4()
    request_id_str = str(request_id_uuid)
    # Используем MODELS_CONFIG из импортированного config.py
    model_version_to_log = MODELS_CONFIG.get(selected_model_id, {}).get("name", selected_model_id)


    data_to_insert = [{
        "request_id": request_id_uuid,
        "user_id": user_id,
        "chat_id": chat_id,
        "message_id": message_id,
        "news_text": news_text,
        "predicted_label": predicted_label,
        "prediction_probability": probability if probability is not None else 0.0,
        "model_version": model_version_to_log,
        "processing_time_ms": processing_time_ms
    }]
    columns = "(request_id, user_id, chat_id, message_id, news_text, predicted_label, prediction_probability, model_version, processing_time_ms)"
    try:
        client.execute(f"INSERT INTO {CH_DB}.requests_log {columns} VALUES", data_to_insert)
        logging.info(f"Запрос {request_id_str} (модель: {model_version_to_log}) залогирован в БД.")
        return request_id_str
    except Exception as e:
        logging.error(f"Ошибка логирования запроса {request_id_str} в БД: {e}", exc_info=True)
        return None

async def log_feedback_to_db(request_id_str: str, user_id: int, user_rating: str):
    client = get_clickhouse_client()
    if not client:
        logging.warning("Нет подключения к БД, логирование фидбека пропускается.")
        return False
    try:
        request_id_uuid = uuid.UUID(request_id_str)
    except ValueError:
        logging.error(f"Неверный формат request_id для UUID: {request_id_str}")
        return False

    data_to_insert = [{
        "request_id": request_id_uuid,
        "user_id": user_id,
        "user_rating": user_rating
    }]
    columns = "(request_id, user_id, user_rating)"
    try:
        client.execute(f"INSERT INTO {CH_DB}.feedback {columns} VALUES", data_to_insert)
        logging.info(f"Фидбек для запроса {request_id_str} (оценка: {user_rating}) залогирован.")
        return True
    except Exception as e:
        logging.error(f"Ошибка логирования фидбека для запроса {request_id_str} в БД: {e}", exc_info=True)
        return False