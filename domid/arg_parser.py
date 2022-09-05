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
    parser.add_argument('--pre_tr', type=float, default=0.5, help="threshold for pretraining")
    parser.add_argument('--L', type=int, default=3, help="number of MC runs")
    parser.add_argument('--nonbinary', action='store_true', default=False, help='assumes nonbinary distribution for the data' )
    return parser


def parse_cmd_args():
    """
    get args from command line
    """
    parser = mk_parser_main()
    args = parser.parse_args()
    if args.acon is None:
        print("\n\n")
        warnings.warn("no algorithm conf specified, going to use default")
        print("\n\n")
    return args
