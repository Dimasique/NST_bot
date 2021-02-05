#Style transfer telegram bot


Бот является итоговым проектом Deep learning school (осень, 2020)

Основные функции бота:

* nst - функция для генерации картинки с помощью улучшенного алгоритма neural style transfer.
  После ввода этой команды бот потребует от вас картинку с контентом, а затем картинку со стилем.
  Как только он получит оба изображения, бот начнет генерировать результат, который сразу же отправится пользователю.
* gan - функция для генерации картинки с помощью CycleGAN. После ввода команды будут предложены
два стиля различных художников; пользователь должен будет выбрать с помощью нажатия на кнопку тот стиль, который он хочет перенести
  на свою картинку. Затем бот будет ожидать саму картинку для переноса стиля на нее. Как только он получит ее, запустится алгорит,
  по завершении которого пользователь получит результат.
  
* skip - команда для сброса текущего состояния. Если вдруг пользователь передумал или случайно нажал не то, он может начать
диалог с ботом заново с помощью ввода этой команды.
  
* help - команда для отображения помощи по функционалу бота.