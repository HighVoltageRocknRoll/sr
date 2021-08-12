import os
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import json
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
from scipy.misc import imresize


class SceneChangeDetector:
    def __init__(self, videos_list):
        self.videos_list = videos_list
        self.scene_changes = {}
        self.hist_block_size = 32
        self.hist_nbins = 64
        self.filtering_kernel = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
        self.dis_thr = 2.0
        self.diff_thr = 0.01
        self.scene_min_frames = 24

    def detect_scene_changes(self):
        print("Detecting scene changes")
        for i in tqdm(range(len(self.videos_list)), total=len(self.videos_list), unit='video'):
            video_fn = self.videos_list[i]
            video = cv2.VideoCapture(video_fn)
            if not video.isOpened():
                print('Can not open ', video_fn, '.')
                continue
            frames_number = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            dif = []
            for j in range(frames_number):
                print('Calculating histogram differences for ',  os.path.split(video_fn)[-1], '. Frame: ', j, end='\r')
                ret, frame = video.read()
                frame_br = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cur_hist = [[np.histogram(frame_br[y * self.hist_block_size:(y + 1) * self.hist_block_size,
                                                   x * self.hist_block_size:(x + 1) * self.hist_block_size],
                                          bins=self.hist_nbins,
                                          range=(0.0, 255.0))[0] for x in range(width // self.hist_block_size)]
                            for y in range(height // self.hist_block_size)]
                if j == 0:
                    hist_diff = [[0 for _ in range(width // self.hist_block_size)]
                                 for _ in range(height // self.hist_block_size)]
                else:
                    hist_diff = [[np.fabs(np.convolve(cur_hist[y][x] - prev_hist[y][x], self.filtering_kernel, 'same'))
                                  for x in range(width // self.hist_block_size)]
                                 for y in range(height // self.hist_block_size)]
                dif.append(np.sum(hist_diff) /
                           (1024 * (width // self.hist_block_size) * (height // self.hist_block_size)))
                prev_hist = cur_hist

            scene_changes = [0]
            for j in range(frames_number):
                indices = [max(0, j - 3), max(0, j - 2), max(0, j - 1), min(frames_number - 1, j + 1),
                           min(frames_number - 1, j + 2), min(frames_number - 1, j + 3)]
                dis = 6 * dif[j] - dif[indices[0]] - dif[indices[1]] - dif[indices[2]] - dif[indices[3]] - \
                      dif[indices[4]] - dif[indices[5]]
                if dis > self.dis_thr and dif[j] > self.diff_thr:
                    scene_changes.append(j - 1)

            scene_changes = [scene_changes[i] for i in range(1, len(scene_changes))
                             if scene_changes[i] - scene_changes[i - 1] > self.scene_min_frames]

            if len(scene_changes) == 0 or frames_number - 1 - scene_changes[-1] > self.scene_min_frames:
                scene_changes.append(frames_number - 1)
            else:
                scene_changes[-1] = frames_number - 1

            self.scene_changes[os.path.split(video_fn)[-1]] = scene_changes

    def save_scene_changes(self, filename):
        with open(filename, 'w') as json_file:
            json.dump(self.scene_changes, json_file)

    def load_scene_changes(self, filename):
        with open(filename, 'r') as json_file:
            self.scene_changes = json.load(json_file)


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def process_video(video_path, scd, writer, args):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print('Can not open ', video, '. Finishing processing')
        return
    frames_number = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    first_scene_frame = 0
    scd_ind = 0
    examples_num = 0
    frames = []
    if args.type == 'full' and not args.crop:
        args.crop = True
        args.crop_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        args.crop_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    for i in tqdm(range(frames_number), total=frames_number, unit='frame'):
        ret, frame = video.read()
        if i == first_scene_frame:
            last_scene_frame = scd[os.path.split(video_path)[-1]][scd_ind]
            scd_ind += 1
            frame_dist = (last_scene_frame - first_scene_frame) // (args.frames_per_scene + 1)
            target_frames = []
            for j in range(1, args.frames_per_scene + 1):
                center = first_scene_frame + j * frame_dist
                for k in range(-args.temporal_radius, args.temporal_radius + 1):
                    target_frames.append(center + k)
        elif i == last_scene_frame:
            first_scene_frame = last_scene_frame + 1
        if i in target_frames:
            frames.append(frame)
        if len(frames) == 2 * args.temporal_radius + 1:
            frames_lr = []
            frame_hr = []
            for k in range(len(frames)):
                if args.crop and (frames[k].shape[0] != args.crop_height or frames[k].shape[1] != args.crop_width):
                    frames[k] = frames[k][:args.crop_height, :args.crop_width]
                lr_h = frames[k].shape[0] // args.scale_factor
                lr_w = frames[k].shape[1] // args.scale_factor
                frames[k] = frames[k][:lr_h * args.scale_factor, :lr_w * args.scale_factor]
                frame_lr = imresize(frames[k], (lr_h, lr_w), interp='bicubic')
                if args.blur:
                    frame_lr = cv2.GaussianBlur(frame_lr, (args.blur_size, args.blur_size), args.blur_sigma,
                                                borderType=cv2.BORDER_REFLECT_101)
                frames_lr.append(cv2.cvtColor(frame_lr, cv2.COLOR_BGR2YUV)[:, :, 0])
                if k == args.temporal_radius:
                    frame_hr = cv2.cvtColor(frames[k], cv2.COLOR_BGR2YUV)[:, :, 0]
            if args.type == 'blocks':
                for y in range(frames_lr[0].shape[0] // args.stride - 1):
                    for x in range(frames_lr[0].shape[1] // args.stride - 1):
                        feature = {}
                        hr_block = frame_hr[y * args.stride * args.scale_factor:
                                            (y * args.stride + args.block_size) * args.scale_factor,
                                            x * args.stride * args.scale_factor:
                                            (x * args.stride + args.block_size) * args.scale_factor]
                        if np.std(hr_block) > args.block_min_std:
                            feature['hr'] = bytes_feature(hr_block.tostring())
                            for k in range(len(frames_lr)):
                                lr_block = frames_lr[k][y * args.stride:y * args.stride + args.block_size,
                                                        x * args.stride:x * args.stride + args.block_size]
                                feature['lr' + str(k)] = bytes_feature(lr_block.tostring())
                            example = tf.train.Example(features=tf.train.Features(feature=feature))
                            writer.write(example.SerializeToString())
                            examples_num += 1
            elif args.type == 'full':
                feature = {}
                for k in range(len(frames_lr)):
                    feature['lr' + str(k)] = bytes_feature(frames_lr[k].tostring())
                feature['hr'] = bytes_feature(frame_hr.tostring())
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                examples_num += 1
            frames = []

    video.release()

    return examples_num


def get_arguments():
    parser = argparse.ArgumentParser(description='Script for converting video to dataset')
    parser.add_argument('--video_folder', default='./video',
                        help='folder with video')
    parser.add_argument('--dataset_folder', default='./dataset',
                        help='folder where to save dataset examples')
    parser.add_argument('--scene_changes', default=None,
                        help='path to file with scene changes or where to write it (default file for writing '
                             'is scene_changes.json)')
    parser.add_argument('--type', default='blocks', choices=['blocks', 'full'],
                        help='dataset type: whether to split images into blocks or use full images')
    parser.add_argument('--frames_per_scene', default=1, type=int,
                        help='number of frames from each scene to add to dataset')
    parser.add_argument('--temporal_radius', default=0, type=int,
                        help='number of previous and next neighboring frames to include for one example')
    parser.add_argument('--scale_factor', default=2, type=int,
                        help='scale factor for low resolution image')
    parser.add_argument('--block_size', default=17, type=int,
                        help='size of blocks to extract from low resolution image')
    parser.add_argument('--stride', default=17, type=int,
                        help='stride between extracted blocks for low resolution image')
    parser.add_argument('--block_min_std', default=8.0, type=float,
                        help='minimum pixel standard deviation for blocks')
    parser.add_argument('--crop', action='store_true',
                        help='Whether to crop images')
    parser.add_argument('--crop_height', default=1800, type=int,
                        help='height of high resolution image, can be additionally cropped to match scale factor')
    parser.add_argument('--crop_width', default=3400, type=int,
                        help='width of high resolution image, can be additionally cropped to match scale factor')
    parser.add_argument('--blur', action='store_true',
                        help='Whether to blur low resolution images')
    parser.add_argument('--blur_size', default=5, type=int,
                        help='Filter size, used to blur low resolution images')
    parser.add_argument('--blur_sigma', default=.1, type=float,
                        help='Gaussian blur sigma, used to blur low resolution images')

    return parser.parse_args()


def main():
    args = get_arguments()

    if not os.path.exists(args.dataset_folder):
        os.mkdir(args.dataset_folder)

    video_list = [os.path.join(args.video_folder, fn) for fn in os.listdir(args.video_folder)]
    scd = SceneChangeDetector(video_list)
    if args.scene_changes is None or not os.path.exists(args.scene_changes):
        scd.detect_scene_changes()
        if args.scene_changes is not None:
            scd.save_scene_changes(args.scene_changes)
        else:
            scd.save_scene_changes('scene_changes.json')
    else:
        scd.load_scene_changes(args.scene_changes)

    print("Preparing dataset")
    examples_num = 0
    writer = tf.python_io.TFRecordWriter(os.path.join(args.dataset_folder, 'dataset.tfrecords'))
    for i in range(len(video_list)):
        print('Processing ', video_list[i], ', ', str(i + 1), '/', str(len(video_list)))
        examples_num += process_video(video_list[i], scd.scene_changes, writer, args)
    print("Dataset prepared. Total number of examples: " + str(examples_num))
    with open(os.path.join(args.dataset_folder, 'dataset_info.txt'), 'w') as dataset_info:
        dataset_info.write(str(examples_num) + '\n')
        dataset_info.write(str(args.scale_factor) + '\n')
        height = args.block_size if args.type == 'blocks' else args.crop_height // args.scale_factor
        width = args.block_size if args.type == 'blocks' else args.crop_width // args.scale_factor
        for i in range(2 * args.temporal_radius + 1):
            dataset_info.write('lr' + str(i) + ',' + str(height) + ',' + str(width) + ',1' + '\n')
        dataset_info.write('hr,' + str(height * args.scale_factor) + ',' + str(width * args.scale_factor) + ',1' + '\n')

    writer.close()


if __name__ == "__main__":
    main()
