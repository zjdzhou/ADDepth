from networks import create_model
from munch import Munch
import yaml
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.transforms import ToPILImage
from math import pi


# 获取加载好的模型权重
def get_gomo_model(weight_path):
    opt = get_gomo_model_opt()
    # 创建模型
    model = create_model(opt)
    # 加载模型权重
    model = model.load_from_checkpoint(weight_path)
    model.eval()
    model.to("cuda:0")
    return model


def get_gomo_model_opt():
    with open(r'D:\mengmeng\MViTDepth_with_gomo\logs\pretrained\tensorboard\default\version_0\hparams.yaml') as cfg_file:
        opt = Munch(yaml.safe_load(cfg_file))
    return opt


def get_gomo_translate_image(image_tensor, model):
    t_phi = torch.tensor(pi)
    # ToPILImage()(image_tensor[0].cpu()).show("orignal image")
    img_real = (image_tensor * 2) - 1
    # ToPILImage()(img_real[0].cpu()).show("img_real")
    img_fake = model.forward(img_real.cuda(0), t_phi.cuda(0))
    # ToPILImage()(img_fake[0].cpu()).show("img fake before restore")
    img_fake = (img_fake + 1) / 2
    # ToPILImage()(img_fake[0].cpu()).show("after restore")
    return img_fake
    # img_fake = ToPILImage()((img_fake[0].cpu() + 1) / 2)
    # img_fake.show("after gan")
