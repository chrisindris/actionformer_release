# Utilities for combining the rgb and flow features.
# TODO: in future, these features should be incorporated into the pipeline rather than ran manually/separately.

import argparse
import os
import numpy as np
import re


def argument_parser():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--operation", type=str, help="operation to run")
    parser.add_argument(
        "-f",
        "--feature_dir",
        "--feature_directory",
        type=str,
        default="/root/models/video_features/output_video-features_thumos_i3d/i3d/",
        help="directory where the features are stored (input directory) and saved to (output directory)",
    )
    parser.add_argument(
        "--file_ext",
        type=str,
        help="the suffix to add to the output files; corresponds to file_ext in ActionFormer config files",
    )
    parser.add_argument(
        "--rgb_feature_dir",
        type=str,
        default=None,
        help="directory where rgb features are stored. Need not be defined if --feature_dir already is.",
    )
    parser.add_argument(
        "--flow_feature_dir",
        type=str,
        default=None,
        help="directory where flow features are stored. Need not be defined if --feature_dir already is.",
    )
    parser.add_argument("--save_dir", type=str, default=None, help="output directory")
    return parser


def get_features(feature_type, feature_dir):
    return sorted(
        list(
            filter(
                lambda i: feature_type in i,
                os.listdir(feature_dir),
            )
        )
    )


def pad(rgb, flow):
    # return features padded with zero to ensure that they have the same dimension along the first axis (ie. make them have same number of frames)
    rgb_pad = max(flow.shape[0] - rgb.shape[0], 0)
    flow_pad = max(rgb.shape[0] - flow.shape[0], 0)
    rgb_padded = np.pad(rgb, ((0, rgb_pad), (0, 0)))
    flow_padded = np.pad(flow, ((0, flow_pad), (0, 0)))
    return rgb_padded, flow_padded


def main(args):
    rgb_feature_dir = args.rgb_feature_dir or args.feature_dir
    flow_feature_dir = args.flow_feature_dir or args.feature_dir

    rgbs = get_features(
        "rgb.npy" if ("r21d" not in args.rgb_feature_dir) else "r21d.npy",
        rgb_feature_dir,
    )
    flows = get_features("flow.npy", flow_feature_dir)

    assert len(rgbs) == len(flows)

    operation = globals()[args.operation]

    for f in range(len(rgbs)):
        rgb = np.load(os.path.join(rgb_feature_dir, rgbs[f]))
        flow = np.load(os.path.join(flow_feature_dir, flows[f]))
        stem = re.sub("_(rgb|flow|r21d).npy", r"", rgbs[f])

        # add the two arrays together
        rgbflow = operation(rgb, flow)

        np.save(
            os.path.join(args.save_dir or args.feature_dir, stem + args.file_ext),
            rgbflow,
        )


def add(rgb, flow):
    return rgb + flow


def concatenate(rgb, flow):
    rgb, flow = pad(rgb, flow)
    return np.concatenate((rgb, flow), axis=1)


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    main(args)
