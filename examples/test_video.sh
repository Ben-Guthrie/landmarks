N_KEYPOINTS=$1
python scripts/test.py --experiment-name video-"$1"pts --train-dataset video --test-dataset video --iteration 146000
