import train, eval
import subprocess
from libs.core import load_config
from pprint import pprint
import yaml
from pathlib import Path
import os
import argparse


def argument_parser():
    parser = argparse.ArgumentParser(
        description="run kfold experiments for ActionFormer."
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="./configs/i5O/videomaev2/i5O_videomaev2_simple-crop.yaml",
        help="The config files for the k experiments will be generated from this one.",
    )

    parser.add_argument(
        "-k",
        "--num_folds",
        type=int,
        default=5,
        help="Number of folds to run",
    )

    return parser


def run_train(cfg_file, cfg_stem):
    """
    Will run a process and print the output to the console.
    The output is also saved to the log files.
    """

    proc = subprocess.Popen(
        [
            "python",
            "./train.py",
            cfg_file,
            "--output",
            "ckpt",
        ]
    )
    try:
        outs, errs = proc.communicate()
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()


def run_eval(cfg_file, ckpt):
    """
    Will run a process and print the output to the console.
    The output is also saved to the log files.
    """

    proc = subprocess.Popen(["python", "./eval.py", cfg_file, ckpt])
    try:
        outs, errs = proc.communicate()
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()


def dump_config(cfg, filename):
    """Write a config to a file."""
    yaml.dump(cfg, open(filename, "w"))


def kfold_config(cfg, fold, num_folds=5):
    """Create a modified config for running the fold."""
    train_split = [str(f) for f in list(range(1, num_folds + 1))]
    val_split = [train_split.pop(fold - 1)]

    cfg["train_split"] = train_split
    cfg["val_split"] = val_split
    cfg["dataset"][
        "json_file"
    ] = "/data/i5O/i5OData/annotations/i5Oannotations-new-5folds-splitinto-all.json"

    # for undercover-right:
    # cfg["dataset"]["num_classes"] = 6

    return cfg


def main(args):
    base_config_path = args.base_config
    base_config_dir = os.path.dirname(base_config_path)
    base_config_stem = Path(base_config_path).stem

    cfg = load_config(base_config_path)

    num_folds = args.num_folds
    for fold in range(1, num_folds + 1):
        # create the config and save it
        cfg = kfold_config(cfg, fold=fold)

        cfg_foldK = os.path.join(
            base_config_dir,
            base_config_stem + "_fold" + str(fold) + ".yaml",
        )
        dump_config(cfg, cfg_foldK)
        print(cfg_foldK)

        cfg_stem = os.path.basename(cfg_foldK).replace(".yaml", "")
        # print(cfg_stem)

        # --- TRAIN ---

        run_train(cfg_foldK, cfg_stem)

        ckpt_path = os.path.join("./ckpt", cfg_stem + "_ckpt")
        print(ckpt_path)

        # -- EVAL ---

        run_eval(cfg_foldK, ckpt_path)


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    main(args)
