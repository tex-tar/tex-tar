import torchvision.transforms as transforms

transforms_apply = transforms.RandomApply(
    [
        transforms.RandomChoice([transforms.RandomApply([transforms.RandomRotation(degrees=20)],p=0.6),
                       transforms.RandomApply([transforms.RandomRotation(degrees=20,fill=255)],p=0.4)]),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomApply([transforms.GaussianBlur(3)],p=0.6),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)],p=0.6),
        transforms.RandomChoice([
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.05), shear=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.05), shear=0.1,fill=255)
        ])
    ]
,p=0.6) 

loader_transform_train = transforms.Compose([
    transforms_apply,
    transforms.Resize((128,96)),
    transforms.ToTensor()
])

loader_transform_val = transforms.Compose([
    transforms.Resize((128,96)),
    transforms.ToTensor()
])
