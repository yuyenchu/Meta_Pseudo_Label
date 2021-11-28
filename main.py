import tensorflow as tf
import argparse
import pathlib
import glob
import os
from os.path import isdir, exists, join, abspath
import time

from utils import *
import config 

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', dest='path', default='./', type=pathlib.Path, help='data path', required=True)

# def display_progress(step, time, loss, total=1000, c = 100):
#     s = step % total
#     tf.print('\r', end='')
#     tf.print(
#         f'{step:6d}['+'='*((s)//c)+('='if (s+1)%total==0 else '>')+' '*((total-s-1)//c)+']', 
#         end=f' Time: {time:3.2f}s, teacher loss: {loss/(s if s>0 else 1):.5f}, student loss: {loss:.5f}', 
#         flush=True
#     )

if __name__ == "__main__":
    args = parser.parse_args()
    
    DATA_PATH = args.path.resolve(strict=True)
    mpl = MPL(DATA_PATH, **config.mnist)
    mpl.fit(15)
    print('done')            
