import argparse


def make_global_parser():
    parser = argparse.ArgumentParser(description='Deep Learning Arguments',
                                     add_help=False)
    parser.add_argument('--parallel',
                        default=False,
                        type=boolean_string,
                        help="Use Data Parallel - If used set cuda:0")
    parser.add_argument('--cuda',
                        type=str,
                        default='cuda:0',
                        help="GPU Device ID")
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help="Number of data loader workers")
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help="Feature Extractor and Classifier LR")
    parser.add_argument('--batch_size',
                        type=int,
                        default=36,
                        help="Batch Size of Train loaders")
    parser.add_argument('--num_epochs',
                        type=int,
                        default=150,
                        help="Training Epochs")
    parser.add_argument('--log_path',
                        type=str,
                        default='log/',
                        help="Logging directory")
    parser.add_argument('--model_path',
                        type=str,
                        default='models/',
                        help="Checkpoint directory")
    return parser


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def make_parser():
    parser = argparse.ArgumentParser(
        description='Domain Adversarial Tangent Learning',
        parents=[make_global_parser()])

    parser.add_argument(
        '--dset',
        type=str,
        default='office',
        help="Dataset choice: office-home,image-clef,office,visda")
    parser.add_argument(
        '--source_dir',
        type=str,
        default=
        '/home/raabc/Database/domain_adaptation/Office-31/images/amazon/',
        help="Source dataset directory")
    parser.add_argument(
        '--target_dir',
        type=str,
        default=
        '/home/raabc/Database/domain_adaptation/Office-31/images/webcam/',
        help="Target dataset directory")
    parser.add_argument('--bottleneck_dim',
                        type=int,
                        default=256,
                        help="Bottleneck Dimension of Feature Extractor")
    parser.add_argument('--subspace_dim',
                        type=int,
                        default=128,
                        help="Subspace Dimension for Tangent Distances")
    parser.add_argument('--num_protos',
                        type=int,
                        default=31,
                        help="Number of Prototypes")
    parser.add_argument('--dlr',
                        type=float,
                        default=0.005,
                        help="Discriminator LR")
    parser.add_argument('--lent',
                        type=float,
                        default=0.05,
                        help="Ent Min Coeff")
    parser.add_argument('--eval_epoch',
                        type=int,
                        default=5,
                        help="Evaluation Cycle")
    parser.add_argument('--method',
                        type=str,
                        default='datl_mpce',
                        help="[datl|datl_pl|jatl]")
    parser.add_argument('--augmentation',
                        type=str,
                        default='normal',
                        help="[normal|dg]")
    parser.add_argument('--save_sia',
                        default=False,
                        type=boolean_string,
                        help="Save Siamese Prototypes as plot")
    parser.add_argument(
        '--save_features',
        default=False,
        type=boolean_string,
        help=
        "Save Siamese Prototypes and additional features without prototypes as plot."
    )
    parser.add_argument('--invert',
                        default=True,
                        type=boolean_string,
                        help="Invert cost function")
    parser.add_argument(
        '--protoparams',
        default=True,
        type=boolean_string,
        help=
        "Fwp of Siamese Prototypes in feature extractor should be gradient tracked"
    )

    args = parser.parse_args()
    # Optimal Parameters
    if "amazon" in args.source_dir and "dslr" in args.target_dir:
        args.lr = 0.0003
        args.dlr = 0.0005
        args.lent = 0.1

    elif "dslr" in args.source_dir and "webcam" in args.target_dir:
        args.lr = 0.0003
        args.dlr = 0.001

    elif "amazon" in args.source_dir and "webcam" in args.target_dir:
        args.dlr = 0.001
        args.lent = 0.05

    elif "dslr" in args.source_dir and "amazon" in args.target_dir:
        args.dlr = 0.001

    elif "webcam" in args.source_dir and "amazon" in args.target_dir:
        args.dlr = 0.001

    return args
