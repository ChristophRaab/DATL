from cv2 import exp
import torch
from torch import nn
import pandas as pd
from torchvision.utils import save_image
from datl.cnn_visualizations.misc_functions import (
    save_image, save_class_activation_images)
from datl.cnn_visualizations.gradcam import GradCam
from torchvision import transforms
from datl.study import office31
from datl.metrics import cosine_similarity
import argparse


class Interpret(nn.Module):

    def __init__(self, fe, classifier, td):
        super(Interpret, self).__init__()
        self.features = fe[0]
        self.td = td
        self.classifier = nn.Sequential(fe[1:], classifier)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_gradcam(model, img, org_img, target_class, file_name_to_export):
    grad_cam = GradCam(model, target_layer=7)
    # Generate cam mask
    cam = grad_cam.generate_cam(img, target_class)
    # Save mask
    save_class_activation_images(org_img, cam, file_name_to_export)
    print('Grad cam completed')


def convert_tooriginal(img):
    # expects img to be [1,3,224,224]
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    org_img = inv_normalize(img.squeeze(0)).detach()
    org_img = transforms.ToPILImage(mode='RGB')(org_img)
    return org_img


def create_interpreable(model, img, target_class, name):
    org_img = convert_tooriginal(img)

    file_name_to_export = name + "_prototype_class_" + str(target_class)
    save_image(org_img, "results/" + file_name_to_export + "_original.png")
    create_gradcam(model, img, org_img, target_class, file_name_to_export)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Explain DATSA Model')

    parser.add_argument('--source',
                        type=str,
                        default='amazon',
                        help="Domain choice: amazon,webcam,dslr")
    parser.add_argument('--target',
                        type=str,
                        default='webcam',
                        help="Domain choice: amazon,webcam,dslr")
    parser.add_argument('--q',
                        type=int,
                        default=5,
                        help="Number of Siamese Translations to be printed")
    args = parser.parse_args()
    accs = []
    plot_src_domain = args.source
    plot_tgt_domain = args.target
    dataset = office31
    for source in dataset:
        source = source.split("/")[-2]
        for target in dataset:
            target = target.split("/")[-2]
            if source != target:
                try:
                    feature_extractor = torch.load(
                        "models/best_datl_sia_mpce_True_fe_" + source + "_" +
                        target + ".pth.tar").to("cpu")
                    classifier = torch.load(
                        "models/best_datl_sia_mpce_True_classifier_" + source +
                        "_" + target + ".pth.tar").to("cpu")
                    discriminator = torch.load(
                        "models/best_datl_sia_mpce_True_tangent_" + source +
                        "_" + target + ".pth.tar").to("cpu")
                    protos_per_domain = int(discriminator.protos.size(0) // 2)

                    with torch.no_grad():
                        sprots = feature_extractor(
                            discriminator.protos[0:31, :].to("cpu"))
                        tprots = feature_extractor(
                            discriminator.protos[31:, :].to("cpu"))
                        dis = cosine_similarity(sprots, tprots)
                        disk, idx = torch.max(dis, dim=0)
                        _, spreds = nn.Softmax(1)(
                            classifier(sprots)).detach().max(1)
                        _, tpreds = nn.Softmax(1)(
                            classifier(tprots)).detach().max(1)
                        acc = discriminator.plabels[31:].cpu() == idx
                        accs.append([
                            source + "_vs_" + target,
                            round(100 * (acc.sum() / protos_per_domain).item(),
                                  2)
                        ])

                    if source == plot_src_domain and target == plot_tgt_domain:
                        model = Interpret(feature_extractor, classifier,
                                          discriminator)
                        model.train()

                        for i in range(args.q):
                            img = discriminator.protos[i, :].unsqueeze(0).to(
                                "cpu")
                            create_interpreable(model, img, i, "source")

                            target_class = int(
                                discriminator.plabels[i + 31].item())
                            img = discriminator.protos[i + 31, :].unsqueeze(
                                0).to("cpu")
                            create_interpreable(
                                model, img, target_class,
                                "target_nearest_prototype_" +
                                str(idx[i].item()))
                except Exception:
                    print("Dataset or Model for " + str(source) + " vs " +
                          str(target) +
                          " combination not available. Train it first!")

    print(pd.DataFrame(accs))