import torch
from torchvision.transforms import ToTensor,Resize,Compose,ToPILImage,Normalize

loader_transform_train = Compose([
    # transforms_apply,
    Resize((128,96)),
    ToTensor()
])

loader_transform_val = Compose([
    Resize((128,96)),
    ToTensor()
])
