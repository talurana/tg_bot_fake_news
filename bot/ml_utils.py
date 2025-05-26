import os
import logging
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from .config import VECTORIZER_PATH, MODELS_CONFIG

vectorizer_instance = None
models_loaded_instances = {}
lemmatizer_instance = None
stop_words_set = None

def load_ml_components():
    global vectorizer_instance, models_loaded_instances, lemmatizer_instance, stop_words_set
    try:
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(f"Векторизатор не найден: {VECTORIZER_PATH}")
        vectorizer_instance = joblib.load(VECTORIZER_PATH)
        logging.info("Векторизатор успешно загружен.")

        for model_id, config_data in MODELS_CONFIG.items():
            if not os.path.exists(config_data["path"]):
                raise FileNotFoundError(f"Файл модели '{model_id}' не найден: {config_data['path']}")
            models_loaded_instances[model_id] = joblib.load(config_data["path"])
            logging.info(f"Модель '{config_data['name']}' ({model_id}) успешно загружена.")

    except Exception as e:
        logging.error(f"Ошибка загрузки ML моделей: {e}", exc_info=True)
        raise

    try:
        lemmatizer_instance = WordNetLemmatizer()
        stop_words_set = set(stopwords.words('english'))
        logging.info("NLTK ресурсы успешно инициализированы.")
    except LookupError as e:
        logging.error(f"Ошибка инициализации NLTK ресурсов: {e}. Убедитесь, что пакеты скачаны (в Dockerfile).")
        raise

def preprocess_text(text: str) -> str:
    if lemmatizer_instance is None or stop_words_set is None:
        logging.error("NLTK компоненты (lemmatizer/stopwords) не инициализированы!")
        return ""
    if not isinstance(text, str):
        logging.warning(f"preprocess_text получил не строку: {type(text)}. Возвращаю пустую строку.")
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    processed_tokens = []
    for word in tokens:
        if word not in stop_words_set and word.isalpha():
            processed_tokens.append(lemmatizer_instance.lemmatize(word))
    return " ".join(processed_tokens)

async def predict_fake_news(news_text: str, model_id: str) -> tuple[str, float | None]:
    if vectorizer_instance is None or not models_loaded_instances:
        logging.error("ML компоненты (vectorizer/models) не загружены!")
        return "Ошибка: ML компоненты не готовы", None

    if model_id not in models_loaded_instances:
        logging.error(f"Запрошена неизвестная модель: {model_id}")
        return "Ошибка: модель не найдена", None
    
    selected_model = models_loaded_instances[model_id]
    model_friendly_name = MODELS_CONFIG[model_id]["name"]

    try:
        preprocessed_text = preprocess_text(news_text)
        if not preprocessed_text.strip():
            return "Не удалось обработать текст", None

        text_vector = vectorizer_instance.transform([preprocessed_text])
        prediction = selected_model.predict(text_vector)[0]
        
        probability = None
        if hasattr(selected_model, "predict_proba"):
            proba = selected_model.predict_proba(text_vector)[0]
            probability = float(max(proba))

        label = "FAKE 🤥" if prediction == 1 else "REAL ✅"
        logging.info(f"Предсказание с помощью '{model_friendly_name}': {label}, {probability}")
        return label, probability
    except Exception as e:
        logging.error(f"Ошибка при предсказании с моделью {model_friendly_name}: {e}", exc_info=True)
        return f"Ошибка предсказания ({model_friendly_name})", None