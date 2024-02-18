# When ~/models/pytorch-i3d-feature-extraction/ is used to extract features, the output are .npz files where <file>['feature'] has shape (1, num_of_starting_frames, 1024).
# However, ActionFormer prefers to have .npy files with shape (num_of_starting_frames, 1024).
# This simple script aids in that conversion.
# TODO: in future, these features should be incorporated into the pipeline rather than ran manually/separately.

import argparse
import os
import numpy as np


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Process some .npz files in a directory."
    )
    parser.add_argument(
        "--input_feature_dir",
        type=str,
        default="/root/models/pytorch-i3d-feature-extraction/out_rgb_imagenet_fps30_oversample_freq4/",
        help="directory containing .npz features produced by ~/models/pytorch-i3d-feature-extraction/",
    )
    parser.add_argument(
        "--output_feature_dir",
        type=str,
        help="directory for where to store the .npy features",
    )
    return parser


def convert(in_feature):
    # given the path, return the contained feature matrix and the video name
    out_feature = np.load(in_feature)
    video_basename = str(out_feature["video_name"])
    out_feature = out_feature["feature"][0, :, :]  # for clarity
    return out_feature, video_basename


def main(args):
    os.makedirs(args.output_feature_dir, exist_ok=True)

    in_features = os.listdir(args.input_feature_dir)
    for in_feature in in_features:
        out_feature, video_basename = convert(
            os.path.join(args.input_feature_dir, in_feature)
        )
        ext = in_feature.replace(video_basename, "").replace(".npz", ".npy")
        np.save(
            os.path.join(args.output_feature_dir, video_basename + ext), out_feature
        )


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    main(args)
