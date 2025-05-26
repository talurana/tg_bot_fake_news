import logging
import time
from aiogram import types, Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command
from aiogram.types import ParseMode


from ..states import NewsAnalysis
from ..keyboards import get_model_choice_keyboard, get_feedback_keyboard, feedback_cb
from ..ml_utils import predict_fake_news, MODELS_CONFIG
from ..clickhouse_utils import log_request_to_db, log_feedback_to_db


async def cmd_analyze_start(message: types.Message, state: FSMContext):
    """
    Starts the news analysis process.
    """
    await state.finish()
    await NewsAnalysis.waiting_for_model_choice.set()
    await message.reply("Пожалуйста, выберите модель для анализа:", reply_markup=get_model_choice_keyboard())

async def process_model_choice_handler(message: types.Message, state: FSMContext):
    """
    Handles the user's model choice.
    """
    chosen_model_name = message.text
    selected_model_id = None
    for model_id, config_data in MODELS_CONFIG.items():
        if config_data["name"] == chosen_model_name:
            selected_model_id = model_id
            break
    
    if not selected_model_id:
        if chosen_model_name.lower() != 'отмена':
            await message.reply("Пожалуйста, выберите модель из предложенных вариантов на клавиатуре.",
                                reply_markup=get_model_choice_keyboard())
        return

    await state.update_data(selected_model_id=selected_model_id)
    await NewsAnalysis.waiting_for_news_text.set()
    await message.reply(f"Выбрана: *{chosen_model_name}*.\n"
                        f"{MODELS_CONFIG[selected_model_id]['description']}\n\n"
                        "Теперь отправьте мне текст новости для анализа.",
                        reply_markup=types.ReplyKeyboardRemove(), parse_mode=ParseMode.MARKDOWN)

async def process_news_text_handler(message: types.Message, state: FSMContext):
    """
    Processes the news text sent by the user.
    """
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

async def process_feedback_callback_handler(callback_query: types.CallbackQuery, callback_data: dict, state: FSMContext):
    request_id = callback_data.get("request_id")
    rating = callback_data.get("rating")
    user_id = callback_query.from_user.id

    logging.info(f"Получен фидбек от user {user_id} для request_id {request_id}: {rating}")
    success = await log_feedback_to_db(request_id, user_id, rating)

    if success:
        await callback_query.answer("Спасибо за ваш отзыв! 😊")
        rating_text = "положительный" if rating == "correct" else "отрицательный"
        try:
            await callback_query.bot.send_message(
                chat_id=callback_query.message.chat.id,
                text=f"Ваш {rating_text} отзыв по запросу учтен. Спасибо!"
            )
        except Exception as e:
            logging.error(f"Не удалось отправить сообщение подтверждения фидбека: {e}", exc_info=True)
        try:
            await callback_query.bot.edit_message_reply_markup(
                chat_id=callback_query.message.chat.id,
                message_id=callback_query.message.message_id,
                reply_markup=None
            )
        except Exception as e:
            logging.warning(f"Не удалось убрать клавиатуру из сообщения после фидбека: {e}")
    else:
        await callback_query.answer("Произошла ошибка при сохранении отзыва. Попробуйте позже.", show_alert=True)

def register_analysis_handlers(dp: Dispatcher):
    dp.register_message_handler(cmd_analyze_start, Command(commands=['analyze']), state="*")
    dp.register_message_handler(process_model_choice_handler, state=NewsAnalysis.waiting_for_model_choice)
    dp.register_message_handler(process_news_text_handler, state=NewsAnalysis.waiting_for_news_text, content_types=types.ContentType.TEXT)
    dp.register_callback_query_handler(process_feedback_callback_handler, feedback_cb.filter(), state="*")