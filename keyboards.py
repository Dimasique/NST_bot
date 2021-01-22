from aiogram import types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import answers

inline_kb_nst = InlineKeyboardMarkup(row_width=2)

text_and_data = (
        (answers.LOAD_STYLE, 'btn_style'),
        (answers.LOAD_CONTENT, 'btn_content'),
    )

row_btns = (types.InlineKeyboardButton(text, callback_data=data) for text, data in text_and_data)

inline_kb_nst.row(*row_btns)