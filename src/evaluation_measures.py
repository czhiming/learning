#-*- coding:utf8 -*-
'''
evaluation_measures -- Additional metrics used in learn_model not implemented in
sklearn.


@author:     Jose' de Souza
        
@copyright:  2012. All rights reserved.
        
@license:    Apache License 2.0

@contact:    jose.camargo.souza@gmail.com
@deffield    updated: Updated
'''
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import manhattan_distances
import numpy as np
from scipy.stats import pearsonr,spearmanr

def pearson_corrcoef(x,  y):
    """
    计算皮尔森相关系数
    """
    
    #print pearsonr(x,y)
    #return " ".join(map(lambda x : str(x), pearsonr(x,  y)))
    return str(pearsonr(x,y)[0])

def mean_absolute_error(x, y):
    """
    MAE 平均绝对误差
    """
    vector = manhattan_distances(np.array([list(x)]),np.array([list(y)]))
    summation = np.sum(vector)
                     
    mae = summation / y.shape[0]
    
    return mae

def root_mean_squared_error(x, y):
    """
    RMSE 均方根误差
    """
    mse = mean_squared_error(x, y)
    rmse = np.sqrt(mse)
    return rmse

##############
# ranking 排序指标
##############
def spearmanr_corrcoef(x,  y):
    """
    计算斯皮尔曼相关系数
    :param x: ndarray
    :param y: ndarray
    """
    return str(spearmanr(x,y)[0])

def DeltaAvg():
    pass

if __name__ == '__main__':
    pass
