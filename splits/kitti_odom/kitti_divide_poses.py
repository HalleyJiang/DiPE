# written by Hualie Jiang, PhD at CUHKSZ, jianghualie0@gmail.com

import argparse
import os
import glob

def devide_poses():
    parser = argparse.ArgumentParser(description='divide kitti odometry poses')
    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI odometry data',
                        required=True)
    opt = parser.parse_args()
    pose_path = os.path.join(opt.data_path, "poses")
    pose_files = glob.glob(pose_path + "/*.txt")
    for pose_file in pose_files:
        seq = pose_file.split('/')[-1].split('.')[0]
        pose_dir = os.path.join(opt.data_path, "sequences", seq, "poses")
        if not os.path.exists(pose_dir):
            os.mkdir(pose_dir)
        with open(pose_file, 'r') as f:
            lines = f.read().splitlines()
        for idx, line in enumerate(lines):
            file_name = os.path.join(pose_dir, "{:06d}.txt".format(idx))
            with open(file_name, 'w') as f:
                f.write(line)


if __name__ == "__main__":
    devide_poses()