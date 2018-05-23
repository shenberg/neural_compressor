import os
import pathlib
from tqdm import tqdm

files = pathlib.Path('/home/shenberg/Documents/compression/valid').glob('*.png')
for file in tqdm(files):
    os.system('python3 encdec.py -c --cuda "{}" "{}"'.format(str(file), '/home/shenberg/Documents/neural_compressor/enc_tmp/' + file.stem + '.enc'))
