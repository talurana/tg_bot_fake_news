import os
import logging
from dotenv import load_dotenv

load_dotenv()

APP_TOKEN = os.getenv("APP_TOKEN")
if not APP_TOKEN:
    logging.critical("Токен бота не найден в .env!")
    raise ValueError("Токен бота не найден в .env!")


CH_HOST = os.getenv("CH_HOST", "clickhouse-server")
CH_PORT = int(os.getenv("CH_PORT", 9000))
CH_USER = os.getenv("CH_USER", "bot_user")
CH_PASSWORD = os.getenv("CH_PASSWORD", "")
CH_DB = os.getenv("CH_DB", "fakenews_db")


MODEL_DIR = "models"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer_new.pkl")
MODELS_CONFIG = {
    "linear_svc": {
        "path": os.path.join(MODEL_DIR, "lsvc_model_new.pkl"),
        "name": "LinearSVC (Быстрая)",
        "description": "Линейный классификатор. Быстрый, хорошо подходит для текста."
    },
    "lgbm": {
        "path": os.path.join(MODEL_DIR, "lgbm_model_new.pkl"),
        "name": "LightGBM (Точная)",
        "description": "Градиентный бустинг. Высокая точность, может быть медленнее."
    }
}


LOGGING_FORMAT = "%(levelname)s: %(asctime)s - %(module)s - %(message)s"
LOGGING_DATE_FORMAT = "%d-%b-%y %H:%M:%S"
LOGGING_LEVEL = logging.INFO

def setup_logging():
    logging.basicConfig(
        format=LOGGING_FORMAT,
        datefmt=LOGGING_DATE_FORMAT,
        level=LOGGING_LEVEL,
    )
