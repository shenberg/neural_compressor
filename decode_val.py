import os
import pathlib
from tqdm import tqdm

files = pathlib.Path('/home/shenberg/Documents/neural_compressor/enc_tmp').glob('*.enc')
for file in tqdm(files):
    os.system('python3 encdec.py -x "{}" "{}"'.format(str(file), '/home/shenberg/Documents/neural_compressor/dec_tmp/' + file.stem + '.png'))
