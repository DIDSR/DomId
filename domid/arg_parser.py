"""
Command line arguments
"""
import argparse
import warnings

from domainlab import arg_parser


def mk_parser_main():
    """
    Args for command line definition
    """
    parser = arg_parser.mk_parser_main()
    parser.add_argument('--d_dim', type=int, default=7,
                        help='number of domains (or clusters)')
    parser.add_argument('--pre_tr', type=int, default=25, help="number of pretraining epochs")
    parser.add_argument('--L', type=int, default=3, help="number of MC runs")
    parser.add_argument('--prior', type = str, default="Bern", help='specifies whether binary or continuous-valued '
                                                                    ' input data.Input either "Bern" for Bernoulli or '
                                                                    '"Gaus" for Gaussian prior distribution for the data.')
    parser.add_argument('--model', type = str, default="linear", help = "specify 'linear' for a fully-connected or "
                                                                        "'cnn' for a convolutional model architecture" )
    parser.add_argument('--inject_var', type=str, default=None, help="name of the injected variable (column) in the csv file")
    parser.add_argument('--meta_data_csv', type=str, default=None,
                        help="path to the csv file containing the meta data for injection "
                             "(use if the file is not in the dataset folder or is not named dataset.csv)")
    parser.add_argument('--dim_inject_y', type=int, default=0, help="dimension to inject to input of the decoder from annotation")


    return parser

def parse_cmd_args():
    """
    Parse given command line arguments
    """
    parser = mk_parser_main()
    args = parser.parse_args()
    return args
