import torch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.markers as mmarkers
from datl.data_loader import load_data


def plot_domains_with_siamodel(source_loader,
                               target_loader,
                               protos,
                               plabels,
                               feature_extractor,
                               args,
                               i=0,
                               name="datl"):
    with torch.no_grad():
        xs, _, ys, yds = load_data(feature_extractor, source_loader, args)
        xt, _, yt, ydt = load_data(feature_extractor,
                                   target_loader,
                                   args,
                                   dl=1)
        xs, ys, yds, xt, yt, ydt = xs.cpu().numpy(), ys.cpu().int().numpy(
        ), yds.cpu().int().numpy(), xt.cpu().numpy(), yt.int().cpu().numpy(
        ), ydt.cpu().int().numpy()
        num_protos = int(protos.size(0) // 2)
        dplabels = np.concatenate(
            [0 * np.ones(num_protos), 1 * np.ones(num_protos)]).astype(int)
        plabels = plabels.detach().cpu().numpy()
        protos = feature_extractor(protos).detach().cpu().numpy()
        pmakers = np.array(["^"] * num_protos + ["v"] * num_protos)
        plot_tsne_prototypes(np.concatenate([xs, xt]),
                             np.concatenate([yds, ydt]),
                             protos,
                             dplabels,
                             name=name + "_protos_domain_" + str(i),
                             pmakers=pmakers,
                             seperate_features=args.save_features)
        plot_tsne_prototypes(np.concatenate([xs, xt]),
                             np.concatenate([ys, yt]),
                             protos,
                             plabels,
                             name=name + "_protos_classification_" + str(i),
                             pmakers=pmakers,
                             seperate_features=args.save_features)


def plot_tsne_prototypes(x,
                         y,
                         px,
                         py,
                         name,
                         cmap=None,
                         dir='plots',
                         pmakers="D",
                         seperate_features=False):
    # x: Array of features from FE
    # y: Ground truth labels for each feature array
    # px: Prototype Array
    # py: Prototype Label Array
    # name: Name for file and type of plot
    # cmap: (optional) Dictionary for color mapping, for ex.: colors_per_class = {0: (255, 241, 0), 1: (255, 140, 0), 2: (232, 17, 35)}

    data = np.concatenate([x, px], 0)
    d = min_max_norm_tsne(data)
    proto_size = 40
    data_size = 2

    make_tsne_prototype_plot(x, y, py, name, dir, pmakers, d, proto_size,
                             data_size)

    if seperate_features:
        make_tsne_plot(y, name, d[:x.shape[0], :])


def min_max_norm_tsne(x):
    tsne = TSNE(n_components=2)

    data = tsne.fit_transform(x)
    data_max, data_min = np.max(data, 0), np.min(data, 0)
    d = (data - data_min) / (data_max - data_min)
    return d


def make_tsne_plot(y, name, d):
    plt.figure()
    if 'classification' in name:
        plt.scatter(d[:, 0], d[:, 1], s=2,
                    c=y.flatten())  #cmap=plt.get_cmap("tab20"))
    else:
        colors = [
            'dodgerblue' if label == 0 else 'darkred'
            for label in y.ravel().tolist()
        ]
        plt.scatter(d[:, 0], d[:, 1], s=2, color=colors)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.savefig(
        "plots/tsne_" + name + ".jpg",
        transparent=True,
        dpi=400,
    )
    plt.savefig("plots/tsne_" + name + ".pdf",
                transparent=True,
                dpi=400,
                bbox_inches='tight',
                pad_inches=0.05)
    plt.close()


def make_tsne_prototype_plot(x, y, py, name, dir, pmakers, d, proto_size,
                             data_size):
    plt.figure()
    if 'classification' in name:
        plt.scatter(d[:x.shape[0], 0],
                    d[:x.shape[0], 1],
                    s=data_size,
                    c=y.flatten(),
                    marker=".")
        mscatter(d[x.shape[0]:, 0],
                 d[x.shape[0]:, 1],
                 s=proto_size,
                 c=py.flatten(),
                 m=pmakers,
                 edgecolors='b')
    else:
        colors = [
            'dodgerblue' if label == 1 else 'darkred'
            for label in y.ravel().tolist()
        ]
        pcolors = [
            'dodgerblue' if label == 1 else 'darkred'
            for label in py.ravel().tolist()
        ]
        plt.scatter(d[:x.shape[0], 0],
                    d[:x.shape[0], 1],
                    s=data_size,
                    color=colors)
        mscatter(d[x.shape[0]:, 0],
                 d[x.shape[0]:, 1],
                 s=proto_size,
                 c=pcolors,
                 m=pmakers,
                 edgecolors='b')

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.savefig(
        str(dir) + "/tsne_prototypes_" + name + ".jpg",
        transparent=True,
        dpi=400,
    )
    plt.savefig(str(dir) + "/tsne_prototypes_" + name + ".pdf",
                transparent=True,
                dpi=400,
                bbox_inches='tight',
                pad_inches=0.05)
    plt.close()


def mscatter(x, y, z=None, ax=None, m=None, **kw):
    ax = ax or plt.gca()

    if z is not None:
        sc = ax.scatter(x, y, z, **kw)
    else:
        sc = ax.scatter(x, y, **kw)

    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc
