import numpy as np
import pandas as pd
# from angara.dda.adversarial.datl._train_datl import train_datl
# from angara.dda.adversarial.datl._train_datl_sia import train_datl as train_datl_sia
from datl.train_datl import train_datl_sia
import torch
from datl.make_config import make_parser

methods = {"datl_mpce": train_datl_sia}

ic = "~/Database/domain_adaptation/imageCLEF/images"
o31 = "~/Database/domain_adaptation/Office-31/images/"

image_clef = [ic + "p/", ic + "i/", ic + "c/"]
office31 = [o31 + "amazon/", o31 + "webcam/", o31 + "dslr/"]


def study_iteration(args, source, target, iters=3):
    args.source_dir = source
    args.target_dir = target
    tmp_acc = []
    for _ in range(iters):
        acc, _ = methods[args.method](args)
        tmp_acc.append(acc)
    return args, tmp_acc


def save_results(args, results):
    results = pd.DataFrame(results)
    results.to_csv("results/study_" + args.method + "_" +
                   str(args.num_protos) + "_" +
                   str(args.source_dir.split("/")[-2]) + "_" +
                   str(args.target_dir.split("/")[-2]) + ".csv")


def do_study(args):
    results = []
    if args.dset == "image-clef":
        dataset = image_clef
    elif args.dset == "office":
        dataset = office31
    else:
        print("dataset not found")

    for source in dataset:
        for target in dataset:
            if source != target:
                args, tmp_acc = study_iteration(args, source, target)

                results.append([np.mean(tmp_acc), np.std(tmp_acc)])
                save_results(args, results)


if __name__ == "__main__":
    args = make_parser()
    do_study(args)

# Examples
#python study.py --cuda cuda:0 --num_protos 31 --method datl_mpce --num_epochs 200 --dset office
