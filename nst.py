from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import torch
from model import Net
from torch.autograd import Variable

IMAGE_SIZE = 256


def load(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)

    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img.unsqueeze(0)


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch


def run(style_path, content_path):
    content = load(content_path)
    style = load(style_path)

    style = preprocess_batch(style)

    style_model = Net()
    model_dict = torch.load('./weights.model')

    style_model.load_state_dict(model_dict, False)

    style_v = Variable(style)

    content_image = Variable(preprocess_batch(content))
    style_model.setTarget(style_v)
    output = style_model(content_image)
    save(output[0])


def save(img):
    (b, g, r) = torch.chunk(img, 3)
    img = torch.cat((r, g, b))

    img = img.clone().clamp(0, 255).detach().numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save('res.jpg')