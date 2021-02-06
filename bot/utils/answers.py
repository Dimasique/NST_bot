START = 'Привет! Я - бот, написанный в качестве итогового проекта Deep learning school. Я умею переносить стиль с ' \
        'одной картинки на другую с помощью нейронных' \
        'сетей. Все доступные тебе команды ты можешь найти в меню, особое внимание обращу на команду /help, ' \
        'она поможет тебе получше разобраться с тем, как со мной работать. '

HELP = r'Для твоего пользования доступны две команды: /nst и /gan. Обе отвечают за перенос стиля, однако, ' \
       r'работают они по-разному, так как в их логике ' \
       r'используются два разных подхода к решению задачи нейронного переноса стиля.' \
       '\n\n' \
       r'Если ты хочешь перенести стиль с помощью команды /nst, то я запрошу у тебя две картинки: одну из них я ' \
       r'использую как основу, а со ' \
       r'второй возьму стиль и перенесу его на первую картинку.' \
       '\n\n' \
       r'Для команды /gan немного другие правила: после ее ввода ты должен будешь выбрать, какой стиль из ' \
       r'предложенных тебе ты хотел бы ' \
       r'видеть на своей картинке. После выбора я попрошу у тебя основную картинку, на которую ты хочешь перенести ' \
       r'стиль.'

CHOOSE_NST = 'Окей, давай начнем! Отправь мне отдельно две фотографии: сначала ту, с которой я возьму стиль, ' \
             'затем ту, которую я использую как основу. '

CHOOSE_GAN = 'Выбери стиль художника, который ты хочешь перенести на свою фотографию.'

CANCEL = 'Окей, давай начнем заново.'

GETTING_IMAGE_CONTENT_ERROR = 'Извини, я не понимаю тебя :(\nЯ жду от тебя фотографию, на которую я перенесу ' \
                              'стиль. '

GETTING_IMAGE_STYLE_ERROR = 'Извини, я не понимаю тебя :(\n Я жду от тебя фотографию, с которой я возьму стиль.'

WAITING_FOR_IMAGE_CONTENT = 'Получил! Теперь последний шаг - отправь мне основную фотографию и жди результат ;)'

WAITING_FOR_IMAGE_GAN = 'Жду фотографию!'

WORKING = 'Приступаю к работе!\nТы можешь продолжать со мной общение, как только я сделаю фотографию, я тебе ее тут ' \
          'же отправлю! '

DONE = 'Готово!'

WRONG_COMMAND = 'Извини, я не понимаю тебя :(\nВведи, команду /help для просмотра доступных команд.'
