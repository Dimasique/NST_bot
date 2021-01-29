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

gan_kb = InlineKeyboardMarkup(row_width=2)
btn_vangogh = InlineKeyboardButton('ван Гог', callback_data='vangogh')
btn_monet = InlineKeyboardButton('Монэ', callback_data='monet')

gan_kb.add(btn_vangogh)
gan_kb.add(btn_monet)


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
    await NST_States.waiting_for_content.set()
    await bot.send_message(message.chat.id, NST_CHOOSE, reply_markup=kb)


@dp.message_handler(state=NST_States.waiting_for_content, content_types=ContentType.ANY)
async def incoming_content_nst(message: types.message, state: FSMContext):
    if len(message.photo) > 0:
        await state.update_data(content=message.photo[-1])
        await NST_States.waiting_for_style.set()
        await bot.send_message(message.chat.id, WAIT_FOR_STYLE, reply_markup=kb)
    else:
        await bot.send_message(message.chat.id, GETTING_IMAGE_ERROR, reply_markup=kb)


@dp.message_handler(state=NST_States.waiting_for_style, content_types=ContentType.ANY)
async def incoming_style_nst(message: types.message, state: FSMContext):
    if len(message.photo) > 0:

        data_dict = await state.get_data()
        style = message.photo[-1]
        content = data_dict['content']

        style_name = f'{style.file_id}.jpg'
        content_name = f'{content.file_id}.jpg'

        await style.download(style_name)
        await content.download(content_name)

        await bot.send_message(message.chat.id, WORKING, reply_markup=kb)

        nst.run(style_name, content_name)
        answer = InputFile(path_or_bytesio='res.jpg')
        await bot.send_photo(message.chat.id, answer, DONE)

        await state.finish()
        os.remove(style_name)
        os.remove(content_name)

    else:
        await bot.send_message(message.chat.id, GETTING_IMAGE_ERROR, reply_markup=kb)


#############################################################################

@dp.message_handler(commands=['gan'], state='*')
async def wrong_message(message: types.message):
    await GAN_States.waiting_for_painter.set()
    await bot.send_message(message.chat.id, GAN_CHOOSE, reply_markup=gan_kb)


@dp.callback_query_handler(lambda c: c.data == 'vangogh', state=GAN_States.waiting_for_painter)
async def process_callback_button1(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await GAN_States.waiting_for_content.set()
    await bot.send_message(callback_query.from_user.id, 'Нажата первая кнопка!')

@dp.callback_query_handler(lambda c: c.data == 'monet', state=GAN_States.waiting_for_painter)
async def process_callback_button2(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await GAN_States.waiting_for_content.set()
    await bot.send_message(callback_query.from_user.id, 'Нажата вторая кнопка!')


@dp.message_handler(state=GAN_States.waiting_for_content, content_types=ContentType.ANY)
async def incoming_content_gan(message: types.message, state: FSMContext):
    if len(message.photo) > 0:

        content = message.photo[-1]
        content_name = f'./GAN/img/{content.file_id}.jpg'

        await content.download(content_name)
        await bot.send_message(message.chat.id, WORKING, reply_markup=kb)

        nst.run_gan(content.file_id, 'style_monet')
        path = './GAN/res_gan/res.jpg'

        answer = InputFile(path_or_bytesio=path)
        await bot.send_photo(message.chat.id, answer, DONE)
        await state.finish()

    else:
        await bot.send_message(message.chat.id, GETTING_IMAGE_ERROR, reply_markup=kb)


###########################################################################################


@dp.message_handler(state="*", content_types=ContentType.ANY)
async def wrong_message(message: types.message):
    await bot.send_message(message.chat.id, WRONG_COMMAND, reply_markup=kb)


async def on_startup(dp):
    await bot.delete_webhook()
    await bot.set_webhook(WEBHOOK_URL)


async def on_shutdown(dp):
    pass


if __name__ == '__main__':
    start_webhook(dispatcher=dp, webhook_path=WEBHOOK_PATH, on_startup=on_startup, on_shutdown=on_shutdown,
                  skip_updates=False, host=WEBAPP_HOST, port=WEBAPP_PORT)
