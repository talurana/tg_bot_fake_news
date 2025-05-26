from aiogram.dispatcher.filters.state import State, StatesGroup

class NewsAnalysis(StatesGroup):
    waiting_for_model_choice = State()
    waiting_for_news_text = State()