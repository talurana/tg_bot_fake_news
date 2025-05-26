from aiogram import Dispatcher

from .common import register_common_handlers
from .analysis import register_analysis_handlers

def register_all_handlers(dp: Dispatcher):
    register_common_handlers(dp)
    register_analysis_handlers(dp)