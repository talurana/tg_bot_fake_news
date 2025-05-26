import logging
from aiogram import types, Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import CommandStart, CommandHelp, Command, Text


async def send_welcome_cmd(message: types.Message, state: FSMContext):
    """
    Handler for /start and /help commands
    """
    await state.finish()
    user_name = message.from_user.first_name
    await message.reply(
        f"Привет, {user_name}! 👋 Я бот для определения фейковых новостей.\n\n"
        "Вот что я умею:\n"
        "👉 /analyze - Начать анализ текста новости.\n"
        "👉 /help - Показать это сообщение еще раз.\n"
        "👉 /cancel - Отменить текущее действие.\n\n"
        "Просто отправь мне команду, чтобы начать!",
        parse_mode=types.ParseMode.MARKDOWN
    )

async def cancel_cmd_handler(message: types.Message, state: FSMContext):
    """
    Handler for /cancel command or "Отмена" text in any state
    """
    logging.info(f"Cancel handler triggered for user {message.from_user.id} with text: '{message.text}'")
    current_state = await state.get_state()
    if current_state is None:
        await message.reply('Нечего отменять.', reply_markup=types.ReplyKeyboardRemove())
        return

    logging.info(f'Cancelling state {current_state} for user {message.from_user.id}')
    await state.finish()
    await message.reply('Действие отменено.', reply_markup=types.ReplyKeyboardRemove())


def register_common_handlers(dp: Dispatcher):
    dp.register_message_handler(send_welcome_cmd, CommandStart(), state="*")
    dp.register_message_handler(send_welcome_cmd, CommandHelp(), state="*")
    dp.register_message_handler(cancel_cmd_handler, Command(commands=['cancel']), state="*")
    dp.register_message_handler(cancel_cmd_handler, Text(equals='отмена', ignore_case=True), state="*")
    