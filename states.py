from aiogram.dispatcher.filters.state import State, StatesGroup


class TestStates(StatesGroup):
    choose_nst = State()
    waiting_for_style_nst = State()
    waiting_for_image_content = State()