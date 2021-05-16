
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder,default_loader,IMG_EXTENSIONS
import copy
import torch

def training_augmentation(resize = 256,crop = 224):
    return transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomResizedCrop(crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def validation_augmentation(resize = 256,crop = 224):
    return transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

