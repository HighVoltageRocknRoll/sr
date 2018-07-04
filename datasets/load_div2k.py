import requests
import os
import argparse
import math
from tqdm import tqdm
import zipfile
import shutil

train_data = ['http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip',
              'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip']
test_data = ['http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip',
             'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip']


def get_arguments():
    parser = argparse.ArgumentParser(description='Script for loading div2k dataset')
    parser.add_argument('--save_folder', default='./', help='folder where to save loaded files')

    return parser.parse_args()


def main():
    args = get_arguments()

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    if not os.path.exists(os.path.join(args.save_folder, 'train')):
        os.mkdir(os.path.join(args.save_folder, 'train'))
    if not os.path.exists(os.path.join(args.save_folder, 'test')):
        os.mkdir(os.path.join(args.save_folder, 'test'))

    for url in train_data:
        print('Processing ', url)
        name = url.split('/')[-1]
        if not os.path.exists(os.path.join(args.save_folder, 'train', name)):
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 2 ** 20
            with open(os.path.join(args.save_folder, 'train', name), "wb") as handle:
                for data in tqdm(response.iter_content(chunk_size=block_size),
                                                       total=math.ceil(total_size // block_size),
                                                       unit='MB', unit_scale=True):
                    handle.write(data)

        print('Unzipping file')
        with zipfile.ZipFile(os.path.join(args.save_folder, 'train', name), "r") as zip_file:
            zip_file.extractall(os.path.join(args.save_folder, 'train'))
        print('Done')

    if os.path.exists(os.path.join(args.save_folder, 'train', 'lr')):
        shutil.rmtree(os.path.join(args.save_folder, 'train', 'lr'))
    os.rename(os.path.join(args.save_folder, 'train', 'DIV2K_train_LR_bicubic'),
              os.path.join(args.save_folder, 'train', 'lr'))
    if os.path.exists(os.path.join(args.save_folder, 'train', 'hr')):
        shutil.rmtree(os.path.join(args.save_folder, 'train', 'hr'))
    os.rename(os.path.join(args.save_folder, 'train', 'DIV2K_train_HR'),
              os.path.join(args.save_folder, 'train', 'hr'))

    for url in test_data:
        print('Processing ', url)
        name = url.split('/')[-1]
        if not os.path.exists(os.path.join(args.save_folder, 'test', name)):
            response = requests.get(url, stream=True)
            name = url.split('/')[-1]

            total_size = int(response.headers.get('content-length', 0))
            block_size = 2 ** 20
            with open(os.path.join(args.save_folder, 'test', name), "wb") as handle:
                for data in tqdm(response.iter_content(chunk_size=block_size),
                                                       total=math.ceil(total_size // block_size),
                                                       unit='MB', unit_scale=True):
                    handle.write(data)
        with zipfile.ZipFile(os.path.join(args.save_folder, 'test', name), "r") as zip_file:
            zip_file.extractall(os.path.join(args.save_folder, 'test'))
        print('Done')

    if os.path.exists(os.path.join(args.save_folder, 'test', 'lr')):
        shutil.rmtree(os.path.join(args.save_folder, 'test', 'lr'))
    os.rename(os.path.join(args.save_folder, 'test', 'DIV2K_valid_LR_bicubic'),
              os.path.join(args.save_folder, 'test', 'lr'))
    if os.path.exists(os.path.join(args.save_folder, 'test', 'hr')):
        shutil.rmtree(os.path.join(args.save_folder, 'test', 'hr'))
    os.rename(os.path.join(args.save_folder, 'test', 'DIV2K_valid_HR'),
              os.path.join(args.save_folder, 'test', 'hr'))


if __name__ == "__main__":
    main()
