import requests
import os
import argparse
import math
import re
from tqdm import tqdm


files_to_download = [
    'https://harmonicinc.box.com/shared/static/wrlzswfdvyprz10hegws74d4wzh7270o.mp4',
    'https://harmonicinc.box.com/shared/static/58pxpuh1dsieye19pkj182hgv6fg4gof.mp4',
    'https://harmonicinc.box.com/shared/static/6uws3kg4ldxtkeg5k5jwubueaolkqsr0.mp4',
    'https://harmonicinc.box.com/shared/static/51ma04aviaeunhzelpw455sodv7judiu.mp4',
    'https://harmonicinc.box.com/shared/static/uaj2o8ku7qhwwzviga9znzviwqg14x1g.mp4',
    'https://harmonicinc.box.com/shared/static/e425git3jtnugqh8llzlgvr0r2j4j351.mp4',
    'https://harmonicinc.box.com/shared/static/29b0z4w9lj4p54q2hf7il9jz6codx36v.mp4',
    'https://harmonicinc.box.com/shared/static/n8x168w6vhpv240hggw7wtj8mszg7wnb.mp4',
    'https://harmonicinc.box.com/shared/static/6inss29is5b7jzxv1qkuf2p9qeaomi04.mp4',
    'https://harmonicinc.box.com/shared/static/v21fqn77ib1r8zlrbnl6fsyzt6rrjj0v.mp4',
    'https://harmonicinc.box.com/shared/static/tmzm8y7bfzpote9obs7le3olh5j87iir.mp4'
]


def get_arguments():
    parser = argparse.ArgumentParser(description='Script for loading and processing video from harmonic')
    parser.add_argument('--save_folder', default='./', help='folder where to save loaded files')
    parser.add_argument('--cut_logo', action='store_true',
                        help='cuts harmonic logo (just crops source video)')
    parser.add_argument('--crop_size', default='3840,2160',
                        help='specifies target width and height, separated by comma')

    return parser.parse_args()


def main():
    args = get_arguments()

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    files_in_save_folder = os.listdir(args.save_folder)
    for url in files_to_download:
        print('Processing ', url)
        response = requests.get(url, stream=True)
        re_results = re.findall('filename=".+"', response.headers.get('content-disposition'))
        if len(re_results) != 0:
            name = re_results[0].split('=')[1][1:-1]
        else:
            name = None
            print('Cant get file name. Skipping.')
        if name is not None and name in files_in_save_folder:
            print('File is already downloaded')
        else:
            print('Downloading...')

            total_size = int(response.headers.get('content-length', 0))
            block_size = 2 ** 20
            with open(os.path.join(args.save_folder, name), "wb") as handle:
                for data in tqdm(response.iter_content(chunk_size=block_size),
                                                       total=math.ceil(total_size // block_size),
                                                       unit='MB', unit_scale=True):
                    handle.write(data)
            print('Done')


if __name__ == "__main__":
    main()
