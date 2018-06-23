import itertools
import random
from pathlib import Path


DATA_PATH = 'D:\WORKSPACE\DATASETS\Defects_Type1_6'
N_SPLIT = 900

train_chain = (list(item.glob('*'))[:N_SPLIT] for item in Path(DATA_PATH).glob('*/*'))
train_list = [item.as_posix() for item in itertools.chain.from_iterable(train_chain)]

test_chain = (list(item.glob('*'))[N_SPLIT:] for item in Path(DATA_PATH).glob('*/*'))
test_list = [item.as_posix() for item in itertools.chain.from_iterable(test_chain)]

random.shuffle(train_list)
random.shuffle(test_list)

with open('train.lst', mode='w+', encoding='utf-8') as f_trn, open('test.lst', mode='w+', encoding='utf-8') as f_tst:
    for item in train_list:
        print(item, file=f_trn)
    for item in test_list:
        print(item, file=f_tst)
