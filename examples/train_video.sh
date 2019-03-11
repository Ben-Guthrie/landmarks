N_KEYPOINTS=$1
python scripts/train.py --configs configs/paths/default.yaml configs/experiments/video-"$1"pts.yaml
