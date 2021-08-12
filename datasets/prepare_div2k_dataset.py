import os
import argparse
from tqdm import tqdm
import cv2
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_arguments():
    parser = argparse.ArgumentParser(description='Script for converting div2k images to dataset')
    parser.add_argument('--div2k_folder', default='./div2k',
                        help='folder with div2k images')
    parser.add_argument('--dataset_folder', default='./dataset',
                        help='folder where to save dataset examples')
    parser.add_argument('--type', default='blocks', choices=['blocks', 'full'],
                        help='dataset type: whether to split images into blocks or use full images')
    parser.add_argument('--scale_factor', default=2, type=int, choices=[2],
                        help='scale factor for low resolution image')
    parser.add_argument('--block_size', default=24, type=int,
                        help='size of blocks to extract from low resolution image')
    parser.add_argument('--stride', default=24, type=int,
                        help='stride between extracted blocks for low resolution image')
    parser.add_argument('--crop_height', default=816, type=int,
                        help='height of high resolution image to crop to when using full images dataset, '
                             'can be additionally cropped to match scale factor')
    parser.add_argument('--crop_width', default=2040, type=int,
                        help='width of high resolution image to crop to when using full images dataset, '
                             'can be additionally cropped to match scale factor')
    parser.add_argument('--temporal_radius', default=0, type=int,
                        help='number of previous and next neighboring frames, used to duplicate low-res samples to '
                             'use dataset for video super-resolution models')

    return parser.parse_args()


def main():
    args = get_arguments()

    if not os.path.exists(args.dataset_folder):
        os.mkdir(args.dataset_folder)

    print("Preparing dataset")
    if args.type != 'blocks':
        args.crop_height = args.crop_height // args.scale_factor * args.scale_factor
        args.crop_width =  args.crop_width // args.scale_factor * args.scale_factor

    examples_num = 0
    writer = tf.python_io.TFRecordWriter(os.path.join(args.dataset_folder, 'dataset.tfrecords'))
    hr_image_fns = os.listdir(os.path.join(args.div2k_folder, 'hr'))
    for i in tqdm(range(len(hr_image_fns)), total=len(hr_image_fns), unit='image'):
        image = cv2.imread(os.path.join(args.div2k_folder, 'hr', hr_image_fns[i]))
        image_lr = cv2.imread(os.path.join(args.div2k_folder, 'lr', 'X' + str(args.scale_factor),
                                           hr_image_fns[i].split('.')[0] + 'x' + str(args.scale_factor) +
                                           '.' + hr_image_fns[i].split('.')[1]))
        if args.type == 'full':
            if image.shape[0] > image.shape[1]:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                image_lr = cv2.rotate(image_lr, cv2.ROTATE_90_CLOCKWISE)
            image = image[:args.crop_height, :args.crop_width]
            image_lr = image_lr[:args.crop_height // args.scale_factor, :args.crop_width // args.scale_factor]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)[:, :, 0]
        image_lr = cv2.cvtColor(image_lr, cv2.COLOR_BGR2YUV)[:, :, 0]
        if args.type == 'blocks':
            height = image_lr.shape[0]
            width = image_lr.shape[1]
            for y in range(height // args.stride - 1):
                for x in range(width // args.stride - 1):
                    lr_block = image_lr[y * args.stride:y * args.stride + args.block_size,
                                        x * args.stride:x * args.stride + args.block_size]
                    hr_block = image[
                            y * args.stride * args.scale_factor:(y * args.stride + args.block_size) * args.scale_factor,
                            x * args.stride * args.scale_factor:(x * args.stride + args.block_size) * args.scale_factor]
                    feature = {'hr': bytes_feature(hr_block.tostring())}
                    for i in range(2 * args.temporal_radius + 1):
                        feature['lr' + str(i)] = bytes_feature(lr_block.tostring())
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                    examples_num += 1
        elif args.type == 'full':
            feature = {'hr': bytes_feature(image.tostring())}
            for i in range(2 * args.temporal_radius + 1):
                feature['lr' + str(i)] = bytes_feature(image_lr.tostring())
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            examples_num += 1

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
