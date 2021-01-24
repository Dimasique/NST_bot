import asyncio
import logging
import os

from aiogram import Bot, types, executor, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.dispatcher import Dispatcher, FSMContext
from aiogram.dispatcher.webhook import SendMessage
from aiogram.utils.executor import start_webhook
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.types import InputFile

from aiogram.types.message import ContentType

from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton

from urllib.parse import urljoin

from answers import *
from states import *

from keyboards import *

import nst

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.environ.get('TOKEN')

WEBHOOK_HOST = os.environ.get('WEBHOOK_HOST')
WEBHOOK_PATH = f'/{BOT_TOKEN}'
WEBHOOK_URL = urljoin(WEBHOOK_HOST, WEBHOOK_PATH)

WEBAPP_HOST = '0.0.0.0'
WEBAPP_PORT = os.environ.get('PORT')

loop = asyncio.get_event_loop()
bot = Bot(token=BOT_TOKEN, loop=loop)

dp = Dispatcher(bot, storage=MemoryStorage())
dp.middleware.setup(LoggingMiddleware())

button_nst = KeyboardButton('/nst')
button_gan = KeyboardButton('/gan')
button_help = KeyboardButton('/help')
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


@dp.message_handler(commands=['cancel'], state="*")
async def cancel(message: types.Message, state: FSMContext):
    await state.finish()
    await bot.send_message(message.chat.id, CANCEL, reply_markup=kb)


# __________________________NST_________________________________________#

@dp.message_handler(commands=['nst'], state="*")
async def choose_nst(message: types.Message):
    await TestStates.waiting_for_content_nst.set()
    await bot.send_message(message.chat.id, NST_CHOOSE, reply_markup=empty_kb)


@dp.message_handler(state=TestStates.waiting_for_content_nst, content_types=ContentType.ANY)
async def incoming_content_nst(message: types.message, state: FSMContext):
    if len(message.photo) > 0:
        await state.update_data(content=message.photo[-1])
        await TestStates.waiting_for_style_nst.set()
        await bot.send_message(message.chat.id, WAIT_FOR_STYLE, reply_markup=kb)
    else:
        await bot.send_message(message.chat.id, 'Что-то не так :(\nПопробуй еще раз', reply_markup=kb)


@dp.message_handler(state=TestStates.waiting_for_style_nst, content_types=ContentType.ANY)
async def incoming_style_nst(message: types.message, state: FSMContext):
    if len(message.photo) > 0:

        data_dict = await state.get_data()
        style = message.photo[-1]
        content = data_dict['content']

        style_name = f'{style.file_id}.jpg'
        content_name = f'{content.file_id}.jpg'

        await style.download(style_name)
        await style.download(content_name)

        await bot.send_message(message.chat.id, WORKING, reply_markup=kb)

        ph = InputFile(path_or_bytesio=style_name)
        await bot.send_photo(message.chat.id, ph)

        #result = nst.run(style_name, content_name)
        #result.save("res.jpg")

        #await bot.send_photo(message.chat.id, 'res.jpg', 'Вуаля!')

        await state.finish()

    else:
        await bot.send_message(message.chat.id, 'Что-то не так :(\nПопробуй еще раз', reply_markup=kb)


#############################################################################


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
