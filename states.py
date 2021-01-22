from aiogram.dispatcher.filters.state import State, StatesGroup


class TestStates(StatesGroup):
    choose_nst = State()
    loading_img = State()