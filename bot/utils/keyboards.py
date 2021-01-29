from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton

kb = ReplyKeyboardMarkup()

button_nst = KeyboardButton('/nst')
button_gan = KeyboardButton('/gan')
button_help = KeyboardButton('/help')
button_cancel = KeyboardButton('/cancel')

kb.add(button_nst)
kb.add(button_gan)
kb.add(button_help)
kb.add(button_cancel)

gan_kb = InlineKeyboardMarkup(row_width=2)
btn_vangogh = InlineKeyboardButton('ван Гог', callback_data='vangogh')
btn_monet = InlineKeyboardButton('Монэ', callback_data='monet')

gan_kb.add(btn_vangogh)
gan_kb.add(btn_monet)
