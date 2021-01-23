from PIL import Image


def upload_img(image_name):
    image = Image.open(image_name)
    return image
