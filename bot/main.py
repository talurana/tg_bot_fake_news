import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from .config import APP_TOKEN, setup_logging
from .clickhouse_utils import get_clickhouse_client
from .ml_utils import load_ml_components
from .handlers import register_all_handlers


setup_logging()

async def on_startup(dp: Dispatcher):
    logging.info("Инициализация ClickHouse клиента...")
    if get_clickhouse_client() is None:
        logging.critical("Не удалось подключиться к ClickHouse. Бот не может стартовать.")
    
    logging.info("Загрузка ML моделей и NLTK ресурсов...")
    try:
        load_ml_components()
    except Exception as e:
        logging.critical(f"Критическая ошибка при загрузке ML компонентов: {e}. Бот не может стартовать.")
        return

    logging.info("Установка команд бота...")
    await set_bot_commands(dp)
    logging.info("Бот готов к работе!")


async def set_bot_commands(dp: Dispatcher):
    await dp.bot.set_my_commands([
        types.BotCommand("start", "🚀 Запустить бота / Помощь"),
        types.BotCommand("analyze", "🔎 Анализировать новость"),
        types.BotCommand("cancel", "❌ Отменить текущее действие"),
    ])

def main():
    logging.info("Запуск бота Fake News Detector...")

    storage = MemoryStorage()
    bot_instance = Bot(token=APP_TOKEN)
    dp = Dispatcher(bot_instance, storage=storage)


    register_all_handlers(dp)
    
    
    executor.start_polling(dp, skip_updates=True, on_startup=on_startup)

if __name__ == "__main__":
    main()