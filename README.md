# Image and video super resolution

This repo contains TensorFlow implementations of following image and video super resolution models:
* SRCNN &mdash; "Image Super-Resolution Using Deep Convolutional Networks" [[arxiv]](https://arxiv.org/abs/1501.00092)
* ESPCN &mdash; "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" [[arxiv]](https://arxiv.org/abs/1609.05158)
* VSRNet &mdash; "Video Super-Resolution With Convolutional Neural Networks" [[ieee]](https://ieeexplore.ieee.org/document/7444187/)
* VESPCN &mdash; "Real-Time Video Super-Resolution with Spatio-Temporal Networks and Motion Compensation" [[arxiv]](https://arxiv.org/abs/1611.05250)

This repo is a part of GSoC project for super resolution filter in ffmpeg.

## Model training

To train provided models you should prepare datasets first using generate_datasets.sh script. It will download several videos (from https://www.harmonicinc.com/4k-demo-footage-download/) to build video dataset for video models and DIV2K dataset (https://data.vision.ee.ethz.ch/cvl/DIV2K/) for image models. After that either of the train scripts for each model can be used to train them. 

## Model generation

To generate binary model files, that can be used in ffmpeg's sr filter, use generate_header_and_model.py script. It additionally produces header files (that are used for internal models in ffmpeg). To use this script specify at least what model to generate and path to the checkpoint files (that can be a folder with several checkpoints, in this case latest checkpoint will be used). For example, to generate model files for trained ESPCN model following command can be used:

    python3 generate_header_and_model.py --model=espcn --ckpt_path=logdir/espcn_batch_32_lr_1e-3_decay_adam/train

## Benchmark results

### Image test set

This test set is produced with generate_datasets.sh script and consists of test part of DIV2K dataset.

Model | PSNR | SSIM | Time (s/image)
----- | :--: | :--: | :------------:
SRCNN | 32.5634 | 0.9234 | 0.1875
ESPCN | 33.8585 | 0.9324 |  0.1532

### Video test set

To get this test set you should manually download HD videos [[Website]](https://media.xiph.org/video/derf/) and use:

    python3 prepare_dataset.py --video_folder=folder_with_videos --dataset_folder=datasets/test --type=full --temporal_radius=1 --frames_per_scene=5 --crop --crop_height=720 --crop_width=1280

Model | PSNR | SSIM | Time (s/image)
----- | :--: | :--: | :------------:
SRCNN | 31.0228 | 0.8914 | 0.0614
ESPCN | 32.0147 | 0.9006 | 0.0492
VSRNet | 30.7809 | 0.8862 | 0.0743
VESPCN | 32.0523 |  0.8991 | 0.0528

