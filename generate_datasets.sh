#!/bin/bash

python3 datasets/load_harmonic_video.py --save_folder=datasets/loaded_harmonic
python3 datasets/load_div2k.py --save_folder=datasets/loaded_div2k

python3 datasets/prepare_dataset.py --video_folder=datasets/loaded_harmonic --dataset_folder=datasets/train --type=blocks --temporal_radius=1 --frames_per_scene=2 --block_size=36 --stride=36 --crop --scene_changes=datasets/scene_changes_harmonic.json --block_min_std=20.0
python3 datasets/prepare_div2k_dataset.py --div2k_folder=datasets/loaded_div2k/train --dataset_folder=datasets/train_div2k --type=blocks --temporal_radius=1 --block_size=36 --stride=36
python3 datasets/prepare_div2k_dataset.py --div2k_folder=datasets/loaded_div2k/test --dataset_folder=datasets/test_div2k --type=full --temporal_radius=1

mkdir datasets/train_merged
cat datasets/train_div2k/dataset.tfrecords datasets/train/dataset.tfrecords > datasets/train_merged/dataset.tfrecords
echo $(($(sed -n 1p datasets/train/dataset_info.txt) + $(sed -n 1p datasets/train_div2k/dataset_info.txt))) > datasets/train_merged/dataset_info.txt
tail -n +2 datasets/train/dataset_info.txt >> datasets/train_merged/dataset_info.txt
