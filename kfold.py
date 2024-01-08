import train, eval
import subprocess
from libs.core import load_config
from pprint import pprint
import yaml
from pathlib import Path
import os


def run_train(fold, cfg_file):
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
            "test_kfold_" + str(fold) + "_splitinto-undercover-right",
        ]
    )
    try:
        outs, errs = proc.communicate()
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()


def run_eval(fold, cfg_file, ckpt):
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
    ] = "/data/i5O/i5OData/annotations/i5Oannotations-5folds-splitinto-undercover-right.json"

    # for undercover-right:
    cfg["dataset"]["num_classes"] = 6

    return cfg


base_config_path = "./configs/i5O_videomaev2_10epochsfinetune.yaml"
base_config_dir = os.path.dirname(base_config_path)
base_config_stem = Path(base_config_path).stem

cfg = load_config(base_config_path)

num_folds = 5
for fold in range(1, num_folds + 1):
    # create the config and save it
    cfg = kfold_config(cfg, fold=fold)

    cfg_foldX = os.path.join(
        base_config_dir,
        base_config_stem
        + "_fold"
        + str(fold)
        + "_splitinto-undercover-right"
        + ".yaml",
    )
    dump_config(cfg, cfg_foldX)

    # -- run the experiment

    run_train(fold, cfg_foldX)

    ckpt_path = os.path.join(
        "./ckpt",
        base_config_stem
        + "_fold"
        + str(fold)
        + "_splitinto-undercover-right"
        + "_test_kfold_"
        + str(fold)
        + "_splitinto-undercover-right",
    )
    run_eval(fold, cfg_foldX, ckpt_path)
