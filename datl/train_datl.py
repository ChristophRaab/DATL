from __future__ import division, print_function

import copy
import time
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datl.data_loader import load_data
from datl.vision_preprocessing import (prototype_preprocessing,
                                       training_augmentation_dg,
                                       validation_augmentation)
from datl.lr_schedule import inv_lr_scheduler, cdan_lda_coeff
from datl.make_config import make_parser
from datl.loss import entropy, mpce
from datl.networks import (create_resnet50_features, grl_hook, init_weights)
from datl.prototypes import SiameseTangentLayer
from datl.helper import setup_logger
from datl.plots import plot_domains_with_siamodel
from kmeans_pytorch import kmeans
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder


def train_datl_sia(args):
    start = time.time()
    name = "datl_sia_mpce_" + str(args.protoparams)
    logging = setup_logger(args, name=name)
    logging.info(str(args))

    source_dataset = ImageFolder(args.source_dir,
                                 transform=training_augmentation_dg())
    target_dataset = ImageFolder(args.target_dir,
                                 transform=training_augmentation_dg())

    validation_dataset = ImageFolder(args.target_dir,
                                     transform=validation_augmentation())
    source_loader = DataLoader(source_dataset,
                               shuffle=True,
                               num_workers=args.num_workers,
                               batch_size=args.batch_size,
                               drop_last=True,
                               pin_memory=True)
    target_loader = DataLoader(target_dataset,
                               shuffle=True,
                               num_workers=args.num_workers,
                               batch_size=args.batch_size,
                               drop_last=True,
                               pin_memory=True)
    validation_loader = DataLoader(validation_dataset,
                                   shuffle=False,
                                   num_workers=args.num_workers,
                                   batch_size=args.batch_size,
                                   pin_memory=True)

    source_init_dataset = ImageFolder(args.source_dir,
                                      transform=prototype_preprocessing())
    source_init_loader = DataLoader(source_init_dataset,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    batch_size=args.batch_size)

    target_init_dataset = ImageFolder(args.target_dir,
                                      transform=prototype_preprocessing())
    target_init_loader = DataLoader(target_init_dataset,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    batch_size=args.batch_size)
    source_loader_size, target_loader_size, validation_loader_size = len(
        source_loader), len(target_loader), len(validation_loader)
    source_dataset_size, target_dataset_size = len(source_dataset), len(
        target_dataset)
    num_classes = len(source_dataset.classes)

    features_extractor = nn.Sequential(
        create_resnet50_features(), nn.Flatten(),
        nn.Linear(2048, args.bottleneck_dim), nn.LeakyReLU(),
        nn.LayerNorm(args.bottleneck_dim)).to(args.cuda)
    classifier = nn.Sequential(nn.Linear(args.bottleneck_dim,
                                         num_classes)).to(args.cuda)
    classifier.apply(init_weights), features_extractor[-1].apply(init_weights)
    features_extractor, classifier = features_extractor.to(
        args.cuda), classifier.to(args.cuda)

    with torch.no_grad():
        sd, sp, sy, _ = load_data(features_extractor, source_init_loader, args)
        td, tp, ty, _ = load_data(features_extractor, target_init_loader, args)

        source_siaprotos = torch.empty(0, 3, 224, 224).to(args.cuda)
        target_siaprotos = torch.empty(0, 3, 224, 224).to(args.cuda)
        source_labels, target_labels = torch.empty(0).to(
            args.cuda), torch.empty(0).to(args.cuda)
        for y in torch.unique(sy):
            idx = torch.where(sy == y)[0]
            selected = idx[torch.randint(idx.shape[0], (1, ))]
            source_siaprotos = torch.cat([source_siaprotos, sp[selected, :]])
            source_labels = torch.cat([source_labels, sy[selected]])

        yt, _ = kmeans(td, num_classes, device=td.device)
        yt = yt.to(args.cuda)
        for y in torch.unique(yt):
            idx = torch.where(yt == y)[0]
            selected = idx[torch.randint(idx.shape[0], (1, ))]
            target_siaprotos = torch.cat([target_siaprotos, tp[selected, :]])
            target_labels = torch.cat([target_labels, ty[selected]])
        siaprotos = torch.cat([source_siaprotos, target_siaprotos])
        plabels = torch.cat([source_labels, target_labels])

    discriminator = SiameseTangentLayer(2, num_classes, args.bottleneck_dim,
                                        args.subspace_dim)
    discriminator = discriminator.to(args.cuda)
    discriminator.init(siaprotos, [sd, td], plabels)
    markers = np.array(["1"] * source_labels.size(0) +
                       ["2"] * target_labels.size(0))
    del sd, td, sp, tp, sy, ty, siaprotos, source_siaprotos, target_siaprotos, target_init_loader
    del source_init_dataset, source_init_loader, target_labels, source_labels, yt, selected, plabels
    torch.cuda.empty_cache()

    optimizer = optim.SGD([{
        'params': features_extractor[:-1].parameters(),
        "lr_mult": 1,
        'decay_mult': 2
    }, {
        'params': features_extractor[-1].parameters(),
        "lr_mult": 10,
        'decay_mult': 2
    }, {
        'params': classifier.parameters(),
        "lr_mult": 10,
        'decay_mult': 2
    }],
                          lr=args.lr,
                          nesterov=True,
                          momentum=0.9,
                          weight_decay=0.0005)
    disop = optim.Adam([{
        'params': discriminator.parameters(),
        "lr_mult": 10,
        'decay_mult': 2
    }],
                       lr=args.dlr,
                       weight_decay=0.0005)  # 0.005 92.0 ohne disop scheduing

    if args.save_sia:
        plot_domains_with_siamodel(source_loader, target_loader,
                                   discriminator.protos, discriminator.plabels,
                                   features_extractor, args, 0, name)

    best_acc, best_model = 0, copy.deepcopy(
        [features_extractor, classifier, discriminator])
    j = 0
    for i in range(args.num_epochs):
        with torch.set_grad_enabled(True):
            avg_loss = avg_acc = avg_dc = classifier_loss = discriminator_loss = loss = 0.0
            training_list = zip(source_loader, cycle(target_loader)) if len(
                source_loader) > len(target_loader) else zip(
                    cycle(source_loader), target_loader)
            training_batches = source_loader_size if source_loader_size > target_loader_size else target_loader_size
            training_size = source_dataset_size if source_dataset_size > target_dataset_size else target_dataset_size
            for (xs, ys), (xt, yt) in training_list:

                xs, ys, xt, yt = xs.to(args.cuda), ys.to(args.cuda), xt.to(
                    args.cuda), yt.to(args.cuda)
                features_extractor.train(), classifier.train(
                ), discriminator.train(),
                optimizer = inv_lr_scheduler(optimizer,
                                             j,
                                             gamma=0.001,
                                             power=0.75)
                disop = inv_lr_scheduler(disop, j, gamma=0.001, power=0.75)
                optimizer.zero_grad(), disop.zero_grad()

                fes = features_extractor(xs)
                fet = features_extractor(xt)

                ls = classifier(fes)
                lt = classifier(fet)

                classifier_loss = nn.CrossEntropyLoss()(ls, ys)

                entropy_loss = entropy(lt)

                yd = torch.from_numpy(
                    np.array([0] * fes.size(0) + [1] * fet.size(0))).long().to(
                        args.cuda)
                fe = torch.cat([fes, fet], dim=0)

                protos = features_extractor(discriminator.protos)
                if args.protoparams:
                    protos = protos.detach()
                yp = torch.from_numpy(
                    np.array([0] * int(protos.size(0) / 2) +
                             [1] * int(protos.size(0) / 2))).long().to(
                                 args.cuda)

                j += 1
                lda = cdan_lda_coeff(j)
                fe.register_hook(grl_hook(lda))
                dis = discriminator(fe, protos, discriminator.subspaces)
                discriminator_loss = mpce(dis,
                                          yd,
                                          yp,
                                          invert_distance=args.invert)

                loss = classifier_loss + discriminator_loss + args.lent * entropy_loss
                loss.backward()
                optimizer.step(), disop.step()

                discriminator.orthogonalize_subspace()
                _, preds = nn.Softmax(1)(ls).detach().max(1)

                avg_loss = avg_loss + loss
                avg_dc = avg_dc + discriminator_loss
                avg_acc = avg_acc + (preds == ys).sum()

        if i % args.eval_epoch == 0:
            with torch.set_grad_enabled(False):

                vavg_loss, vavg_acc = 0.0, 0.0
                for xt, yt in validation_loader:

                    xt, yt = xt.to(args.cuda), yt.to(args.cuda)
                    features_extractor.eval(), classifier.eval(
                    ), discriminator.eval()

                    lt = classifier(features_extractor(xt))

                    _, preds = nn.Softmax(1)(lt).max(1)
                    classifier_loss = nn.CrossEntropyLoss()(lt, yt)

                    loss = classifier_loss

                    vavg_loss = vavg_loss + loss
                    vavg_acc = vavg_acc + (preds == yt).sum()

                vavg_acc = (vavg_acc / target_dataset_size).item()
                if best_acc < vavg_acc:
                    best_acc, best_model = vavg_acc, copy.deepcopy(
                        [features_extractor, classifier, discriminator])
                logging.info(
                    "Progress " + str(i) + ", " + str(j) +
                    ", Mean Validation Loss: " +
                    str(round((vavg_loss /
                               validation_loader_size).item(), 3)) +
                    ", Acc :" + str(round(vavg_acc, 3)) +
                    " --- Mean Training Loss: " +
                    str(round((avg_loss / training_batches).item(), 3)) +
                    ", Acc :" +
                    str(round((avg_acc / training_size).item(), 3)) +
                    ", DC :" + str(
                        round((avg_dc / (source_loader_size +
                                         target_loader_size)).item(), 2)))
            if args.save_sia:
                plot_domains_with_siamodel(source_loader, target_loader,
                                           discriminator.protos,
                                           discriminator.plabels,
                                           features_extractor, args, j, name)
    torch.save(
        best_model[0], args.model_path + "best_" + name + "_fe_" +
        str(args.source_dir.split("/")[-2]) + "_" +
        str(args.target_dir.split("/")[-2]) + ".pth.tar")
    torch.save(
        best_model[1], args.model_path + "best_" + name + "_classifier_" +
        str(args.source_dir.split("/")[-2]) + "_" +
        str(args.target_dir.split("/")[-2]) + ".pth.tar")
    torch.save(
        best_model[-1], args.model_path + "best_" + name + "_tangent_" +
        str(args.source_dir.split("/")[-2]) + "_" +
        str(args.target_dir.split("/")[-2]) + ".pth.tar")
    duration = round((time.time() - start) / 60, 2)
    logging.info(
        "Finished in " + str(duration) + " minutes with Best Acc" +
        str(best_acc) +
        "========================================================================================"
    )
    return best_acc, best_model


if __name__ == "__main__":

    args = make_parser()
    train_datl_sia(args)
