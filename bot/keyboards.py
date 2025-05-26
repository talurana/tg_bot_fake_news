from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils.callback_data import CallbackData
from .config import MODELS_CONFIG


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
    for model_id, config_data in MODELS_CONFIG.items():
        keyboard.add(KeyboardButton(config_data["name"]))
    keyboard.add(KeyboardButton("Отмена"))
    return keyboard