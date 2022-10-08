import random
import numpy as np
import os

def na_percent(df):
	print(df.isna().sum()/df.shape[0]*100)


def seed_everything(seed = 123):
	random.seed(seed)
	np.random.seed(seed)
	return seed

def set_dir():
    os.chdir('/home/jpietraszek_new/SVM')
    print('cwd:', os.getcwd())
