import os
from os.path import join, basename
import random
import glob
folders = [("labeled",700),("testing",300)]
os.rename('./training', './labeled')
os.mkdir('./unlabeled')
for dir, num in folders:
    for i in range(10):
        fs = glob.glob(join(dir,str(i),'*.jpg'))
        random.shuffle(fs)
        print(join(dir,str(i)),len(fs))
        for f in fs[num:]:
            os.rename(f, join('unlabeled',f'{dir}_{i}_{basename(f)}'))