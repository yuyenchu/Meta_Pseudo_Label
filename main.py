import tensorflow as tf
import argparse
import pathlib
import glob
import os
from os.path import isdir, exists, join, abspath
import time

from utils import *
from config import config

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', dest='path', default='./data', type=pathlib.Path, help='path to dataset')
parser.add_argument('-s','--save', dest='save', default='./save', type=str, help='path to data output, if not valid, a new folder will be created based on datetime (will be overridden if provided by config)')
parser.add_argument('-e','--epoch', dest='epoch', default=1, type=int, help='number of epochs to train')
parser.add_argument('-c','--config', dest='cfg', default='mnist', type=str, help='name of config to be used, must be provided in config.py')

if __name__ == "__main__":
    args = parser.parse_args()
    
    DATA_PATH = args.path.resolve(strict=True)
    EPOCH = args.epoch
    if EPOCH < 1:
        raise ValueError(f'Training epoch must be an integer greater than zero')
    CONFIG = args.cfg
    if CONFIG not in config:
        raise ValueError(f'The provided config name \'{CONFIG}\' is not valid')
    SAVE_PATH = args.save
    if not (exists(SAVE_PATH) and isdir(SAVE_PATH)):
        print(f'The provided save path \'{SAVE_PATH}\' is not valid, a new folder will be created')
        SAVE_PATH = ''
    mpl = MPL(DATA_PATH, **{'save_path': SAVE_PATH, **config[CONFIG]})
    mpl.fit(EPOCH)
    mpl.save()
    print('-'*27,'Task Done','-'*27)            
