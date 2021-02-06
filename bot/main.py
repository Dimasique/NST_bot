import asyncio
import logging
import os

from queue import Queue
from threading import Thread
import time

from aiogram import Bot, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.dispatcher import Dispatcher, FSMContext
from aiogram.utils.executor import start_webhook
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.types import InputFile

from aiogram.types.message import ContentType

from urllib.parse import urljoin

from utils.answers import *
from utils.states import *
from utils.keyboards import *

import style_transfer

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

task_queue = Queue()
result_queue = Queue()


@dp.message_handler(commands=['start'], state="*")
async def start(message: types.Message):
    await message.answer(START, reply_markup=kb)


@dp.message_handler(commands=['help'], state="*")
async def help(message: types.Message):
    await message.answer(HELP)


@dp.message_handler(commands=['nst'], state="*")
async def choose_nst(message: types.Message):
    await NST_States.waiting_for_content.set()
    await message.answer(CHOOSE_NST)


@dp.message_handler(commands=['gan'], state="*")
async def choose_gan(message: types.message):
    await GAN_States.waiting_for_painter.set()
    await message.answer(CHOOSE_GAN, reply_markup=gan_kb)


@dp.message_handler(commands=['cancel'], state="*")
async def cancel(message: types.Message, state: FSMContext):
    await state.finish()
    await message.answer(CANCEL, reply_markup=kb)


# __________________________NST_________________________________________#


@dp.message_handler(state=NST_States.waiting_for_content, content_types=ContentType.ANY)
async def incoming_content_nst(message: types.message, state: FSMContext):
    if len(message.photo) > 0:
        await state.update_data(content=message.photo[-1])
        await NST_States.waiting_for_style.set()
        await message.answer(WAITING_FOR_IMAGE_STYLE)
    else:
        await message.answer(GETTING_IMAGE_CONTENT_ERROR)


@dp.message_handler(state=NST_States.waiting_for_style, content_types=ContentType.ANY)
async def incoming_style_nst(message: types.message, state: FSMContext):
    if len(message.photo) > 0:
        await message.answer(WORKING)
        data = await state.get_data()
        style = message.photo[-1]
        content = data['content']

        style_name = f'bot/images/{style.file_id}.jpg'
        content_name = f'bot/images/{content.file_id}.jpg'

        await style.download(style_name)
        await content.download(content_name)

        task = {'id': message.chat.id, 'type': 'nst', 'content': content.file_id, 'style' : style.file_id}
        task_queue.put(task)

        #style_transfer.run_nst(style_name, content_name)
        #answer = InputFile(path_or_bytesio='bot/result/res.jpg')
        #await bot.send_photo(message.chat.id, answer, DONE)

        await state.finish()

    else:
        await message.answer(GETTING_IMAGE_STYLE_ERROR)


# _______________________________________________________________________#


# __________________________GAN_________________________________________#


@dp.callback_query_handler(lambda c: c.data == 'vangogh', state=GAN_States.waiting_for_painter)
async def process_callback_vangogh(callback_query: types.CallbackQuery, state: FSMContext):
    await bot.answer_callback_query(callback_query.id)
    await GAN_States.waiting_for_content.set()
    await bot.send_message(callback_query.from_user.id, WAITING_FOR_IMAGE_GAN)
    await state.update_data(model='style_vangogh')


@dp.callback_query_handler(lambda c: c.data == 'monet', state=GAN_States.waiting_for_painter)
async def process_callback_monet(callback_query: types.CallbackQuery, state: FSMContext):
    await bot.answer_callback_query(callback_query.id)
    await GAN_States.waiting_for_content.set()
    await bot.send_message(callback_query.from_user.id, WAITING_FOR_IMAGE_GAN)
    await state.update_data(model='style_monet')


@dp.message_handler(state=GAN_States.waiting_for_content, content_types=ContentType.ANY)
async def incoming_content_gan(message: types.message, state: FSMContext):
    if len(message.photo) > 0:

        content = message.photo[-1]
        content_name = f'bot/images/{content.file_id}.jpg'

        await content.download(content_name)
        await message.answer(WORKING)

        data = await state.get_data()

        task = {'id': message.chat.id, 'type': 'gan', 'content': content.file_id, 'model': data['model']}
        task_queue.put(task)

        # style_transfer.run_gan(content.file_id, data['model'])
        # path = f'bot/result/res.jpg'

        # answer = InputFile(path_or_bytesio=path)
        # await bot.send_photo(message.chat.id, answer, DONE)
        await state.finish()

    else:
        await bot.send_message(message.chat.id, GETTING_IMAGE_CONTENT_ERROR, reply_markup=kb)


# _______________________________________________________________________#


@dp.message_handler(state="*", content_types=ContentType.ANY)
async def wrong_message(message: types.message):
    await message.answer(WRONG_COMMAND, reply_markup=kb)


async def on_startup(dp):
    await bot.delete_webhook()
    await bot.set_webhook(WEBHOOK_URL)


async def on_shutdown(dp):
    pass


def send_result(id):
    photo_res = InputFile(path_or_bytesio='bot/result/res.jpg')
    bot.send_photo(id, photo_res, DONE)


def process_queue(task_queue):
    while True:
        if not task_queue.empty():
            task = task_queue.get()

            if task['type'] == 'nst':
                style_transfer.run_nst(task['style'], task['content'])

            else:
                style_transfer.run_gan(task['content'], task['model'])

            send_fut = asyncio.run_coroutine_threadsafe(send_result(task['id']), loop)
            send_fut.result()
            task_queue.task_done()

        time.sleep(2)


if __name__ == '__main__':
    worker = Thread(target=process_queue, args=(task_queue,))
    worker.start()

    start_webhook(dispatcher=dp, webhook_path=WEBHOOK_PATH, on_startup=on_startup, on_shutdown=on_shutdown,
                  skip_updates=False, host=WEBAPP_HOST, port=WEBAPP_PORT)
