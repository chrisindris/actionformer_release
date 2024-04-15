# Utilities for combining the rgb and flow features.
# TODO: in future, these features should be incorporated into the pipeline rather than ran manually/separately.

import argparse
import os
import numpy as np
import re


def argument_parser():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--operation", type=str, nargs="+", help="operation to run")
    return parser


def main(args):
    operation = globals()[args.operation[0]]
    print(operation(*args.operation[1:]))


def add(xargs=None):
    return "add " + str(xargs)


def concatenate():
    return "concatenate"


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    main(args)
