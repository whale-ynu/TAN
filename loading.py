import re
import copy
import numpy as np
from scipy.stats import kendalltau
import math

def load():
    fpath=open('data_demo/path.txt')
    data_path=fpath.readlines()
    dic={}
    longest = 0
    for line in data_path:
        tmp=line.strip().split()
        u=tmp[0]
        s=tmp[-1]
        if str(u)+' '+str(s) not in dic:
            dic[str(u)+' '+str(s)]=tmp
            long = len(tmp)
            if long >= longest:
                longest = long
    return longest,dic

if __name__ == '__main__':
    load()

