from PIL import Image
#import torchvision.models as models


def upload_img(image_name):
    image = Image.open(image_name)
    return image


def upload_vgg():
    #cnn = models.vgg19(pretrained=True).eval()
    return True
