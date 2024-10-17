import torch
from torchvision.transforms import RandomHorizontalFlip, RandomResizedCrop


def crop_numpy(image, size=(32, 32), scale=(0.6, 1.0), ratio=(1.0, 1.0)):
    tmp = torch.tensor(image)
    flipper = RandomHorizontalFlip(0.5)
    cropper = RandomResizedCrop(size=size, scale=scale, ratio=ratio)
    image = flipper(cropper(tmp))
    image = image.numpy()
    return image