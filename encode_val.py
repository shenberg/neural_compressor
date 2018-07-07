import os
import pathlib
from tqdm import tqdm
import argparse


def main():
    files = pathlib.Path('/home/shenberg/Documents/compression/valid').glob('*.png')
    for file in tqdm(files):
        os.system('python3 encdec.py -c --cuda "{}" "{}"'.format(str(file), '/media/shenberg/ssd_large/results/context/enc_tmp/' + file.stem + '.enc'))

if __name__ == '__main__':
    main()