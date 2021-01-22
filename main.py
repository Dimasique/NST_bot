import asyncio
import logging
import os

from aiogram import Bot, types, executor, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.dispatcher import Dispatcher, FSMContext
from aiogram.dispatcher.webhook import SendMessage
from aiogram.utils.executor import start_webhook
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from aiogram.types.message import ContentType

from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton

from urllib.parse import urljoin

from aiohttp import ClientSession
from answers import *
from states import *

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.environ.get('TOKEN')  # '1512424491:AAGdEuGa_LMUwuijAA4IV6y7_DQztPoOmeE'

WEBHOOK_HOST = os.environ.get('WEBHOOK_HOST')  # 'https://glacial-beyond-67935.herokuapp.com/'
WEBHOOK_PATH = f'/{BOT_TOKEN}'
WEBHOOK_URL = urljoin(WEBHOOK_HOST, WEBHOOK_PATH)

WEBAPP_HOST = '0.0.0.0'
WEBAPP_PORT = os.environ.get('PORT')

loop = asyncio.get_event_loop()
bot = Bot(token=BOT_TOKEN, loop=loop)
dp = Dispatcher(bot, storage=MemoryStorage())

button_nst = KeyboardButton(r'/nst')
button_gan = KeyboardButton(r'/gan')
button_help = KeyboardButton(r'/help')
button_cancel = KeyboardButton('/cancel')

kb = ReplyKeyboardMarkup()

kb.add(button_nst)
kb.add(button_gan)
kb.add(button_help)
kb.add(button_cancel)


@dp.message_handler(commands=['start'], state="*")
async def hello(message: types.Message):
    await bot.send_message(message.chat.id, HELLO, reply_markup=kb)


@dp.message_handler(commands=['help'], state="*")
async def help(message: types.Message):
    await bot.send_message(message.chat.id, HELP, reply_markup=kb)


#__________________________NST_________________________________________#

@dp.message_handler(commands=['nst'], state="*")
async def choose_nst(message: types.Message):
    await TestStates.choose_nst.set()

    inline_kb = InlineKeyboardMarkup(row_width=2)
    inline_kb.add(InlineKeyboardButton('Загрузить картинку со стилем', callback_data='btn_style'))
    inline_kb.add(InlineKeyboardButton('Загрузить картинку со контентом', callback_data='btn_content'))


    await bot.send_message(message.chat.id, NST_CHOOSE, reply_markup=inline_kb)



@dp.callback_query_handler(lambda c: c.data == 'btn_style')
async def process_callback_button1(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(
            callback_query.id, 'Жду стиль!')

@dp.callback_query_handler(lambda c: c.data == 'btn_content')
async def process_callback_button1(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(
        callback_query.id, 'Жду контент!')


@dp.message_handler(state=TestStates.choose_nst)
async def choose_nst_(message: types.Message, state: FSMContext):
    res = 'Получил!' if message.photo is not None else 'Что-то не так :('
    await bot.send_message(message.chat.id, res, reply_markup=kb)
    await state.finish()




#############################################################################

@dp.message_handler(commands=['cancel', 'gan'])
async def dummy(message: types.Message):
    await bot.send_message(message.chat.id, DUMMY, reply_markup=kb)


@dp.message_handler()
async def wrong(message: types.Message, state: FSMContext):
    await bot.send_message(message.chat.id, WRONG_COMMAND, reply_markup=kb)


async def on_startup(dp):
    await bot.delete_webhook()
    await bot.set_webhook(WEBHOOK_URL)
    # insert code here to run it after start


async def on_shutdown(dp):
    # insert code here to run it before shutdown
    pass


if __name__ == '__main__':
    start_webhook(dispatcher=dp, webhook_path=WEBHOOK_PATH, on_startup=on_startup, on_shutdown=on_shutdown,
                  skip_updates=False, host=WEBAPP_HOST, port=WEBAPP_PORT)
