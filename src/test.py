#-*- coding:utf8 -*-
'''
Created on Oct 26, 2016

@author: czm
'''
from sklearn.metrics.pairwise import manhattan_distances
import numpy as np


a = np.array([1,2,3])
b = np.array([2,1,3])

print manhattan_distances(np.array([list(a)]),np.array([list(b)]))













if __name__ == '__main__':
    pass