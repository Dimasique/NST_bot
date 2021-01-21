import asyncio
import logging
import os

from aiogram import Bot, types, executor, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.dispatcher import Dispatcher
from aiogram.dispatcher.webhook import SendMessage
from aiogram.utils.executor import start_webhook
from urllib.parse import urljoin

from aiohttp import ClientSession
from answers import *

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.environ.get('TOKEN') #'1512424491:AAGdEuGa_LMUwuijAA4IV6y7_DQztPoOmeE'

WEBHOOK_HOST = os.environ.get('WEBHOOK_HOST') #'https://glacial-beyond-67935.herokuapp.com/'
WEBHOOK_PATH = f'/{BOT_TOKEN}'
WEBHOOK_URL = urljoin(WEBHOOK_HOST, WEBHOOK_PATH)

WEBAPP_HOST = '0.0.0.0'
WEBAPP_PORT = os.environ.get('PORT')

loop = asyncio.get_event_loop()
bot = Bot(token=BOT_TOKEN, loop=loop)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def hello(message: types.Message):
    await bot.send_message(message.chat.id, HELLO)

@dp.message_handler()
async def help(commands=['help']):
    await bot.send_message(message.chat.id, HELP)

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