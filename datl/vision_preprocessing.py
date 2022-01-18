from torchvision import transforms
import torch


def validation_augmentation(resize=256, crop=224):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def prototype_preprocessing(resize=256, crop=224):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def validation_10crop_augmentation(resize=256, crop=224):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.TenCrop(crop),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            (transforms.ToTensor()(crop)) for crop in crops
        ]))  # returns a 4D tensor
    ])


def training_augmentation_dg(resize=256, crop=224):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomResizedCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])