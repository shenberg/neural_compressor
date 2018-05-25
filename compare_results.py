import pathlib
from PIL import Image
import ms_ssim_baseline
import numpy as np

src_dir = pathlib.Path('/home/shenberg/Documents/compression/valid/')
enc_dir = pathlib.Path('/home/shenberg/Documents/neural_compressor/enc_tmp/')
dec_dir = pathlib.Path('/home/shenberg/Documents/neural_compressor/dec_tmp/')

def compare_dirs(src_dir, enc_dir, dec_dir):
    total_encoded_size = 0
    total_pixels = 0
    similarities = []
    for src_file in src_dir.glob('*png'):
        print("Comparing file {}".format(src_file))
        enc_file = enc_dir / (src_file.with_suffix('.enc').name)
        dec_file = dec_dir / src_file.name
        src_img = Image.open(str(src_file))
        enc_size = enc_file.stat().st_size * 8 # in bits
        pixels = src_img.size[0] * src_img.size[1]
        total_encoded_size += enc_size
        total_pixels += pixels
        similarity = ms_ssim_baseline.msssim(np.array(src_img), str(dec_file))
        similarities.append(similarity)
        print('{}, {}'.format(similarity, enc_size / pixels))
    print('done, totals: (avg ssim, avg br)')
    print('{}, {}'.format(sum(similarities) / len(similarities), total_encoded_size / total_pixels))

compare_dirs(src_dir, enc_dir, dec_dir)