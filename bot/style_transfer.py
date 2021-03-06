from PIL import Image
import numpy as np
import torch
from NST.model import Net
import os
import gc

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


def run_nst(style_path, content_path):
    content = load(content_path).requires_grad_(False)
    style = load(style_path).requires_grad_(False)

    style = preprocess_batch(style)

    style_model = Net()
    model_dict = torch.load('./NST/weights.model')

    style_model.load_state_dict(model_dict, False)
    style_model.eval()

    content_image = preprocess_batch(content)
    style_model.set_gram(style)
    output = style_model(content_image)
    save(output[0])

    gc.collect()
    del style_model
    os.remove(style_path)
    os.remove(content_path)


def save(img):
    (b, g, r) = torch.chunk(img, 3)
    img = torch.cat((r, g, b))

    img = img.clone().clamp(0, 255).detach().numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save('bot/result/res.jpg')


def run_gan(img, model):
    os.system(f'python ./GAN/test.py --dataroot ./bot/images/ --name {model}_pretrained --model test --no_dropout')
    os.remove(f'bot/images/{img}.jpg')
