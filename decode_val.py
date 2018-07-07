import os
import pathlib
from tqdm import tqdm

files = pathlib.Path('/media/shenberg/ssd_large/results/context/enc_tmp').glob('*.enc')
for file in tqdm(files):
    os.system('python3 encdec.py -x --cuda "{}" "{}"'.format(str(file), '/media/shenberg/ssd_large/results/context/dec_tmp/' + file.stem + '.png'))
