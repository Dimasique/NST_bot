from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import answers

inline_kb_nst = InlineKeyboardMarkup(row_width=2)

btn_style = InlineKeyboardButton(answers.LOAD_STYLE, callback_data='btn_style')
btn_content = InlineKeyboardButton(answers.LOAD_CONTENT, callback_data='btn_content')
inline_kb_nst.add(btn_style)
inline_kb_nst.add(btn_content)