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

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.environ['BOT_TOKEN']

WEBHOOK_HOST = os.environ['WEBHOOK_HOST_ADDR']
WEBHOOK_PATH = f'/webhook/{BOT_TOKEN}'
WEBHOOK_URL = urljoin(WEBHOOK_HOST, WEBHOOK_PATH)

WEBAPP_HOST = '0.0.0.0'
WEBAPP_PORT = os.environ['PORT']

bot = Bot(token=BOT_TOKEN)
session = ClientSession()
dp = Dispatcher(bot)
dp.setup_middleware(LoggingMiddleware())


@dp.message_handler()
async def echo(message: types.Message):

    return SendMessage(message.chat.id, message.text)


async def on_startup(dispatcher: 'Dispatcher') -> None:
    logging.warning('Starting...')


async def on_shutdown(dispatcher: 'Dispatcher') -> None:
    logging.warning('Bye!')


if __name__ == '__main__':
    start_webhook(dispatcher=dp,
                  webhook_path=WEBHOOK_PATH,
                  on_startup=on_startup,
                  on_shutdown=on_shutdown,
                  skip_updates=False,
                  host=WEBAPP_HOST,
                  port=WEBAPP_PORT)
