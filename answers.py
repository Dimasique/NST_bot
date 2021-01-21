HELLO = 'Привет! Я - бот, написанный в качестве итогового проекта Deep learning school. Я умею переносить стиль с одной картинки на другую с помощью нейронных' \
        ' сетей. Все доступные тебе команды ты можешь найти в меню, особое внимание обращу на команду \help, она поможет тебе получше разобраться с тем, как со мной работать.'



HELP = r"""
    Для твоего пользования доступны две комманды: \nst и \gan. Обе отвечают за перенос стиля, однако, работают они по-разному, так как
    в их логике используются два разных подхода к решению задачи нейронного переноса стиля.
    
    Если ты хочешь перенести стиль с помощью команды \nst, то я запрошу у тебя две картинки: одну из них я использую как основу, а со
    второй возьму стиль и перенесу его на первую картинку.
    
    Для команды \gan немного другие правила: после ее ввода ты должен будешь выбрать, какой стиль из предложенных тебе ты хотел бы
    видеть на своей картинке. После выбора я попрошу у тебя основную картинку, на которую ты хочешь перенести стиль.
    
    P.S. Не пугайся, если я  буду долго работать. Кажется, работа в районе 10 минут для такого бота, как я, считается нормой.
"""