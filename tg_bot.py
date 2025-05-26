import logging
import os
import uuid
import time
from dotenv import load_dotenv
import joblib
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ParseMode, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils.callback_data import CallbackData
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage # Для хранения состояния

from clickhouse_driver import Client

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


load_dotenv()
logging.basicConfig(
    format="%(levelname)s: %(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)

APP_TOKEN = os.getenv("APP_TOKEN")
if not APP_TOKEN:
    logging.critical("Токен бота не найден!")
    raise ValueError("Токен бота не найден!")


storage = MemoryStorage()
bot = Bot(token=APP_TOKEN)
dp = Dispatcher(bot, storage=storage)

class NewsAnalysis(StatesGroup):
    waiting_for_model_choice = State()
    waiting_for_news_text = State()

CH_HOST = os.getenv("CH_HOST", "clickhouse-server")
CH_PORT = int(os.getenv("CH_PORT", 9000))
CH_USER = os.getenv("CH_USER", "bot_user")
CH_PASSWORD = os.getenv("CH_PASSWORD", "")
CH_DB = os.getenv("CH_DB", "fakenews_db")
db_client = None

def get_clickhouse_client():
    global db_client
    reconnect_needed = False

    if db_client is None:
        reconnect_needed = True
    else:
        try:
            db_client.execute("SELECT 1")
        except Exception as e:
            logging.warning(f"Соединение с ClickHouse потеряно или недействительно: {e}. Попытка переподключения.")
            reconnect_needed = True
            if db_client and db_client.connection:
                try:
                    db_client.disconnect()
                except Exception:
                    pass
            db_client = None

    if reconnect_needed:
        try:
            logging.info(f"Попытка подключения к ClickHouse ({CH_HOST}:{CH_PORT}, БД: {CH_DB})...")
            db_client = Client(
                host=CH_HOST, port=CH_PORT, user=CH_USER, password=CH_PASSWORD, database=CH_DB,
                connect_timeout=10, send_receive_timeout=300,
                settings={'max_block_size': 100000}
            )
            db_client.execute("SELECT 1")
            logging.info(f"Успешное новое подключение к ClickHouse ({CH_HOST}:{CH_PORT}, БД: {CH_DB})")
        except Exception as e:
            logging.error(f"Ошибка нового подключения к ClickHouse: {e}")
            db_client = None

    return db_client


get_clickhouse_client()


MODEL_DIR = "models"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer_new.pkl")
MODELS_CONFIG = {
    "model_1": {
        "path": os.path.join(MODEL_DIR, "lsvc_model_new.pkl"),
        "name": "LinearSVC (Быстрая)",
        "description": "Линейный классификатор. Быстрый, хорошо подходит для текста."
    },
    "model_2": {
        "path": os.path.join(MODEL_DIR, "lgbm_model_new.pkl"),
        "name": "LightGBM (Точная)",
        "description": "Градиентный бустинг. Высокая точность, может быть медленнее."
    }
}
models_loaded = {}
vectorizer = None

try:
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Векторизатор не найден: {VECTORIZER_PATH}")
    vectorizer = joblib.load(VECTORIZER_PATH)
    logging.info("Векторизатор успешно загружен.")

    for model_id, config in MODELS_CONFIG.items():
        if not os.path.exists(config["path"]):
            raise FileNotFoundError(f"Файл модели '{model_id}' не найден: {config['path']}")
        models_loaded[model_id] = joblib.load(config["path"])
        logging.info(f"Модель '{config['name']}' ({model_id}) успешно загружена.")

except Exception as e:
    logging.error(f"Ошибка загрузки моделей: {e}")
    raise


try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    logging.info("NLTK ресурсы успешно инициализированы.")
except LookupError as e:
    logging.error(f"Ошибка инициализации NLTK ресурсов: {e}. Убедитесь, что пакеты скачаны.")
    raise

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        logging.warning(f"preprocess_text получил не строку: {type(text)}. Возвращаю пустую строку.")
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    processed_tokens = []
    for word in tokens:
        if word not in stop_words and word.isalpha():
            processed_tokens.append(lemmatizer.lemmatize(word))
    return " ".join(processed_tokens)



async def predict_fake_news(news_text: str, model_id: str) -> tuple[str, float | None]:
    if model_id not in models_loaded:
        logging.error(f"Запрошена неизвестная модель: {model_id}")
        return "Ошибка: модель не найдена", None
    
    selected_model = models_loaded[model_id]
    model_friendly_name = MODELS_CONFIG[model_id]["name"]

    try:
        preprocessed_text = preprocess_text(news_text)
        if not preprocessed_text.strip():
            return "Не удалось обработать текст", None

        text_vector = vectorizer.transform([preprocessed_text])
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



async def log_request_to_db(user_id: int, chat_id: int, message_id: int,
                            news_text: str, predicted_label: str,
                            probability: float | None, processing_time_ms: int,
                            selected_model_id: str) -> str | None:
    client = get_clickhouse_client()
    if not client:
        logging.warning("Нет подключения к БД, логирование запроса пропускается.")
        return None

    request_id_uuid = uuid.uuid4()
    request_id = str(request_id_uuid)
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
        logging.info(f"Запрос {request_id} (модель: {model_version_to_log}) залогирован в БД.")
        return request_id
    except Exception as e:
        logging.error(f"Ошибка логирования запроса {request_id} в БД: {e}", exc_info=True)
        return None


async def log_feedback_to_db(request_id: str, user_id: int, user_rating: str): # request_id приходит как строка
    client = get_clickhouse_client()
    if not client:
        logging.warning("Нет подключения к БД, логирование фидбека пропускается.")
        return False

    try:
        request_id_uuid = uuid.UUID(request_id)
    except ValueError:
        logging.error(f"Неверный формат request_id для UUID: {request_id}")
        return False

    data_to_insert = [{
        "request_id": request_id_uuid,
        "user_id": user_id,
        "user_rating": user_rating
    }]

    columns = "(request_id, user_id, user_rating)"

    try:
        client.execute(f"INSERT INTO {CH_DB}.feedback {columns} VALUES", data_to_insert)
        logging.info(f"Фидбек для запроса {request_id} (оценка: {user_rating}) залогирован.")
        return True
    except Exception as e:
        logging.error(f"Ошибка логирования фидбека для запроса {request_id} в БД: {e}", exc_info=True)
        return False

feedback_cb = CallbackData("feedback", "request_id", "rating")

def get_feedback_keyboard(request_id: str) -> InlineKeyboardMarkup:
    keyboard = InlineKeyboardMarkup(row_width=2)
    keyboard.add(
        InlineKeyboardButton("👍 Правильно", callback_data=feedback_cb.new(request_id=request_id, rating="correct")),
        InlineKeyboardButton("👎 Ошибка", callback_data=feedback_cb.new(request_id=request_id, rating="incorrect"))
    )
    return keyboard


def get_model_choice_keyboard() -> ReplyKeyboardMarkup:
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    for model_id, config in MODELS_CONFIG.items():
        keyboard.add(KeyboardButton(config["name"]))
    keyboard.add(KeyboardButton("Отмена"))
    return keyboard

async def set_default_commands(dp: Dispatcher):
    await dp.bot.set_my_commands([
        types.BotCommand("start", "🚀 Запустить бота / Помощь"),
        types.BotCommand("analyze", "🔎 Анализировать новость"),
        types.BotCommand("cancel", "❌ Отменить текущее действие"),
    ])

@dp.message_handler(commands=["start", "help"], state="*")
async def send_welcome(message: types.Message, state: FSMContext):
    await state.finish()
    user_name = message.from_user.first_name
    await message.reply(
        f"Привет, {user_name}! 👋 Я бот для определения фейковых новостей.\n\n"
        "Вот что я умею:\n"
        "👉 /analyze - Начать анализ текста новости (я предложу выбрать модель).\n"
        "👉 /help - Показать это сообщение еще раз.\n"
        "👉 /cancel - Отменить текущее действие (например, выбор модели или ввод текста).\n\n"
        "Просто отправь мне команду, чтобы начать!",
        parse_mode=ParseMode.MARKDOWN
    )

@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(lambda message: message.text.lower() == 'отмена', state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        await message.reply('Нечего отменять.', reply_markup=types.ReplyKeyboardRemove())
        return

    logging.info(f'Cancelling state {current_state} for user {message.from_user.id}')
    await state.finish()
    await message.reply('Действие отменено.', reply_markup=types.ReplyKeyboardRemove())

@dp.message_handler(commands=["analyze"], state="*")
async def cmd_analyze(message: types.Message, state: FSMContext):
    await state.finish()
    await NewsAnalysis.waiting_for_model_choice.set()
    await message.reply("Пожалуйста, выберите модель для анализа:", reply_markup=get_model_choice_keyboard())

@dp.message_handler(state=NewsAnalysis.waiting_for_model_choice)
async def process_model_choice(message: types.Message, state: FSMContext):
    chosen_model_name = message.text
    selected_model_id = None
    for model_id, config in MODELS_CONFIG.items():
        if config["name"] == chosen_model_name:
            selected_model_id = model_id
            break
    
    if not selected_model_id:
        await message.reply("Пожалуйста, выберите модель из предложенных вариантов на клавиатуре.",
                            reply_markup=get_model_choice_keyboard())
        return

    await state.update_data(selected_model_id=selected_model_id)
    await NewsAnalysis.waiting_for_news_text.set()
    await message.reply(f"Выбрана: *{chosen_model_name}*.\n"
                        f"{MODELS_CONFIG[selected_model_id]['description']}\n\n"
                        "Теперь отправьте мне текст новости для анализа.",
                        reply_markup=types.ReplyKeyboardRemove(), parse_mode=ParseMode.MARKDOWN)

@dp.message_handler(state=NewsAnalysis.waiting_for_news_text, content_types=types.ContentType.TEXT)
async def process_news_text(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    selected_model_id = user_data.get("selected_model_id")
    news_text = message.text

    if not selected_model_id:
        logging.error(f"В состоянии waiting_for_news_text отсутствует selected_model_id для user {message.from_user.id}")
        await message.reply("Произошла ошибка, модель не была выбрана. Пожалуйста, начните сначала с /analyze.")
        await state.finish()
        return

    start_time = time.time()
    logging.info(f"User {message.from_user.id} (модель: {selected_model_id}) sent text: '{news_text[:100]}...'")

    status_message = await message.reply("🔎 Анализирую новость...")

    label, probability = await predict_fake_news(news_text, selected_model_id)
    processing_time_ms = int((time.time() - start_time) * 1000)

    request_id = await log_request_to_db(
        user_id=message.from_user.id, chat_id=message.chat.id, message_id=message.message_id,
        news_text=news_text, predicted_label=label, probability=probability,
        processing_time_ms=processing_time_ms, selected_model_id=selected_model_id
    )

    response_text = f"Результат ({MODELS_CONFIG[selected_model_id]['name']}): *{label}*"
    if probability is not None:
        response_text += f"\nУверенность: *{probability*100:.2f}%*"
    
    reply_markup = get_feedback_keyboard(request_id) if request_id else None
    
    try:
        await status_message.edit_text(response_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    except Exception:
        await message.reply(response_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)


    await state.finish()


@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(lambda message: message.text.lower() == 'отмена', state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        await message.reply('Нечего отменять.', reply_markup=types.ReplyKeyboardRemove())
        return

    logging.info(f'Cancelling state {current_state} for user {message.from_user.id}')
    await state.finish()
    await message.reply('Действие отменено.', reply_markup=types.ReplyKeyboardRemove())



@dp.callback_query_handler(feedback_cb.filter(), state="*")
async def process_feedback_callback(callback_query: types.CallbackQuery, callback_data: dict, state: FSMContext):
    request_id = callback_data.get("request_id")
    rating = callback_data.get("rating")
    user_id = callback_query.from_user.id

    logging.info(f"Получен фидбек от user {user_id} для request_id {request_id}: {rating}")
    success = await log_feedback_to_db(request_id, user_id, rating)

    if success:
        await callback_query.answer("Спасибо за ваш отзыв! 😊")

        rating_text = "положительный" if rating == "correct" else "отрицательный"
        try:
            await bot.send_message(
                chat_id=callback_query.message.chat.id,
                text=f"Ваш {rating_text} отзыв по запросу учтен. Спасибо!"
            )
        except Exception as e:
            logging.error(f"Не удалось отправить сообщение подтверждения фидбека: {e}", exc_info=True)
        
        try:
            await bot.edit_message_reply_markup(
                chat_id=callback_query.message.chat.id,
                message_id=callback_query.message.message_id,
                reply_markup=None
            )
        except Exception as e:
            logging.warning(f"Не удалось убрать клавиатуру из сообщения после фидбека: {e}")

    else:
        await callback_query.answer("Произошла ошибка при сохранении отзыва. Попробуйте позже.", show_alert=True)


if __name__ == "__main__":
    logging.info("Запуск бота...")
    executor.start_polling(dp, skip_updates=True, on_startup=set_default_commands)
