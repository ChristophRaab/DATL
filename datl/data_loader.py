import torch


def load_data(features_extractor,
              loader,
              args,
              raw_data_shape=(3, 224, 224),
              dl=0):
    embedded_features = torch.empty([0, args.bottleneck_dim]).to(args.cuda)
    raw_features = torch.empty((0, ) + raw_data_shape).to(args.cuda)
    class_labels = domain_labels = torch.empty([0]).to(args.cuda)
    for x, y in loader:
        x, y = x.to(args.cuda), y.to(args.cuda)
        embedded_features = torch.cat(
            [embedded_features, features_extractor(x)], dim=0)
        raw_features = torch.cat([raw_features, x], dim=0)
        class_labels = torch.cat([class_labels, y], dim=0)
        domain_labels = torch.cat([
            domain_labels,
            dl * torch.ones(x.shape[0], dtype=torch.int).to(args.cuda)
        ],
                                  dim=0)
    return embedded_features, raw_features, class_labels, domain_labels
