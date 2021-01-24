from PIL import Image
from model import *

import copy


def upload_img(image_name):
    image = Image.open(image_name)
    return image


def run(style_path, content_path):

    style = upload_img(style_path)
    content = upload_img(content_path)

    model_nst = copy.deepcopy(model)

    style1 = model_nst[1](model_nst[0](style))
    model_nst[2].target = gram_matrix(style1).detach()
    model_nst[2].loss = F.mse_loss(model_nst[2].target, model_nst[2].target)

    style2 = model_nst[4](model_nst[3](model_nst[2](style1)))
    model_nst[5].target = gram_matrix(style2).detach()
    model_nst[5].loss = F.mse_loss(model_nst[5].target, model_nst[5].target)

    style3 = model_nst[8](model_nst[7](model_nst[6](model_nst[5](style2))))
    model_nst[9].target = gram_matrix(style3).detach()
    model_nst[9].loss = F.mse_loss(model_nst[9].target, model_nst[9].target)

    style4 = model_nst[11](model_nst[10](model_nst[9](style3)))

    style5 = model_nst[15](model_nst[14](model_nst[13](style4)))
    model_nst[16].target = gram_matrix(style5).detach()
    model_nst[16].loss = F.mse_loss(model_nst[16].target, model_nst[16].target)

    content4 = model_nst[11](model_nst[10](
        model_nst[9](model_nst[8](model_nst[7](model_nst[6](
            model_nst[5](model_nst[4](model_nst[3](model_nst[2](model_nst[1](model_nst[0](content))))))))))))
    model_nst[12].target = content4.detach()
    model_nst[12].loss = F.mse_loss(model_nst[12].target, model_nst[12].target)

    epoch_num = 500
    style_weight = 200000
    content_weight = 1

    # model, style_losses, content_losses = self.build_model(cnn, style, content)
    # print(model)
    # torch.save(model.state_dict(), 'model_nst.pt')
    style_losses = [model[2], model[5], model[9], model[16]]
    content_losses = [model[12]]

    input_img = content.clone()
    optimizer = optimizer = optim.LBFGS([input_img.requires_grad_()])
    run_ = [0]

    while run_[0] <= epoch_num:

        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()

            model(input_img)

            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run_[0] += 1
                return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img
