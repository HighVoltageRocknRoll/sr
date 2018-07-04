#!/bin/bash

python3 datasets/load_harmonic_video.py --save_folder=datasets/loaded_harmonic --cut_logo
python3 datasets/load_div2k.py --save_folder=datasets/loaded_div2k

python3 datasets/prepare_dataset.py --video_folder=datasets/loaded_harmonic --dataset_folder=datasets/train --type=blocks --temporal_radius=1 --frames_per_scene=7 --block_size=36 --stride=36
python3 datasets/prepare_div2k_dataset.py --div2k_folder=datasets/loaded_div2k/train --dataset_folder=datasets/train_div2k --type=blocks
python3 datasets/prepare_div2k_dataset.py --div2k_folder=datasets/loaded_div2k/test --dataset_folder=datasets/test_div2k --type=full
