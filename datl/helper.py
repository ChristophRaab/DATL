import logging

import torch


def setup_logger(args, name="datl"):
    level = logging.INFO
    format = '  %(message)s'
    [
        logging.root.removeHandler(handler)
        for handler in logging.root.handlers[:]
    ]
    dataset = str(args.source_dir.split("/")[-2]) + "_" + str(
        args.target_dir.split("/")[-2])
    handlers = [
        logging.FileHandler(args.log_path + "log_" + name + "_" + dataset +
                            ".txt"),
        logging.StreamHandler()
    ]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    return logging
