"""
Command line arguments
"""

from domainlab import arg_parser
import numpy as np


def mk_parser_main():
    """
    Args for command line definition
    """
    parser = arg_parser.mk_parser_main()
    parser.add_argument("--d_dim", type=int, default=7, help="number of domains (or clusters)")
    parser.add_argument("--pre_tr", type=int, default=25, help="number of pretraining epochs")
    parser.add_argument("--L", type=int, default=3, help="number of MC runs")
    parser.add_argument(
        "--prior",
        type=str,
        default="Bern",
        help="specifies whether binary or continuous-valued input data. Input either 'Bern' for Bernoulli or 'Gaus' for Gaussian prior distribution for the data.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        help="specify 'linear' for a fully-connected or 'cnn' for a convolutional model architecture",
    )
    parser.add_argument(
        "--inject_var", type=str, default=None, help="name of the injected variable (column) in the csv file"
    )
    parser.add_argument(
        "--meta_data_csv",
        type=str,
        default=None,
        help="path to the csv file containing the meta data for injection (use if the file is not in the dataset folder or is not named dataset.csv)",
    )
    parser.add_argument(
        "--dim_inject_y", type=int, default=0, help="dimension to inject to input of the decoder from annotation"
    )
    parser.add_argument(
        "--digits_from_mnist",
        nargs="*",
        type=int,
        default=None,
        help="digits that should be included from mnist dataset",
    )
    parser.add_argument("--path_to_results", type=str, default="./", help="path to the results csv file")
    parser.add_argument("--pre_tr_weight_path", type=str, default=None, help="path to the pre-trained weights")
    parser.add_argument(
        "--tr_d_range",
        nargs="*",
        default=None,
        help="range to determine the domains used for training; for example, tr_d_range 0 3 assigns domains 0, 1, 2 to training",
    )
    parser.add_argument(
        "--graph_method", type=str, default=None, help="graph calculation method can be specified for SDCN"
    )
    parser.add_argument("--feat_extract", type=str, default="vae", help="featue extractor method, either vae or ae")
    parser.add_argument(
        "--random_batching",
        type=bool,
        default=False,
        help="randomization of the samples inside one batch, only used in SDCN",
    )

    return parser


def parse_cmd_args():
    """
    Parse given command line arguments
    """
    parser = mk_parser_main()
    args = parser.parse_args()
    if args.tr_d_range is not None:
        tr_d_range = np.arange(int(args.tr_d_range[0]), int(args.tr_d_range[1]), 1)
        tr_d_range = [str(i) for i in tr_d_range]
        setattr(args, "tr_d", list(tr_d_range))
    return args
