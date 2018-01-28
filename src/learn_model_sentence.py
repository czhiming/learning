#-*- coding:utf8 -*-
'''
Created on Mar 5, 2017

@author: czm
'''
import sklearn
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy
import time
#from evaluation_measures import pearson_corrcoef,mean_squared_error,mean_absolute_error,spearmanr_corrcoef
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestRegressor
import logging as log

def read_x(file_name,separator):
    """
    读取特征 X
    """
    x = []
    with open(file_name) as fp:
        for lines in fp:
            lines = lines.strip().split(separator)
            lines = map(lambda x:float(x),lines)
            x.append(lines)
    return numpy.array(x)

def read_y(file_name):
    """
    读取标签 y
    """
    y = []
    with open(file_name) as fp:
        for lines in fp:
            lines = lines.strip()
            lines = float(lines)
            y.append(lines)
    return numpy.array(y)

def get_index(i,y_hat):
    """
    获得索引号
    """
    y_hat_ = sorted(enumerate(y_hat),key=lambda x:x[1])
    for key,value in enumerate(y_hat_):
        if i == value[0]:
            return key+1

def optimize_model(estimator, X_train, y_train, params, scores, folds, verbose, n_jobs):
    clf = None
    log.debug(params)
    log.debug(scores)    
    for score_name in scores:
        log.info("Tuning hyper-parameters for %s" % score_name)        
        clf = GridSearchCV(estimator, params, 
                           cv=folds, verbose=verbose, n_jobs=n_jobs)
        
        clf.fit(X_train, y_train)
        
        log.info("Best parameters set found on development set:")
        log.info(clf.best_params_)
        
    return clf.best_estimator_

def prepare_data(x_train,y_train,x_test,y_test,separator='\t',scale=True,flag=None):
    if y_test is not None:
        X_train = read_x(x_train,separator)
        y_train = read_y(y_train)
        X_test = read_x(x_test,separator)
        y_test = read_y(y_test)
        
        ss = StandardScaler()
        if scale:
            X_train = ss.fit_transform(X_train)
            X_test = ss.fit_transform(X_test)
        
        if flag == 'wmt16':
            y_train = y_train/100
            y_test = y_test/100
        
    if y_test is None:
        X_train = read_x(x_train,separator)
        y_train = read_y(y_train)
        X_test = read_x(x_test,separator)

        if flag == 'wmt16':
            y_train = y_train/100

    return X_train,y_train,X_test,y_test

def train(
    x_train = None,
    y_train = None,
    x_test = None,
    y_test = None,
    separator = '\t',
    predict_file = 'predicted.csv',
    ref_file = 'ref.csv',
    team_name = 'JXNU/word2vec+rnnlm',
    parameters = {},
    cv = 5,
    flag = 'wmt15',
    scale = True,
    scores = ['Pearson']
    ):
    start_time = time.time()
    print '设置的参数：',locals()
    log.basicConfig(level=log.DEBUG)
    
    clf = SVR(kernel='rbf',C=1.0,epsilon=0.2)
    #clf = RandomForestRegressor()
    
    if y_test is not None:
        X_train,y_train,X_test,y_test = prepare_data(x_train,y_train,x_test,y_test,separator=separator,scale=scale,flag=None)
        log.info('training ...')
        #是否需要优化模型参数
        if parameters == {}:
            pass
        else:
            clf = optimize_model(clf,X_train,y_train,parameters,scores,cv,True,-1)
        #训练模型
        log.info('Done.')
        log.info('Predict ...')
        y_pred = clf.predict(X_test)
        log.info('Done.')
        if flag == 'wmt16':
            y_pred = y_pred*100
            y_test = y_test*100
        log.info('make result file ...')
        with open(predict_file, 'w') as _fout:
            for i, _y in enumerate(zip(y_test, y_pred)):
                print >> _fout,  "%s\t%d\t%f\t%d" % (team_name,i+1,_y[1],get_index(i,y_pred))
        with open(ref_file,'w') as _fout:
            for i, _y in enumerate(zip(y_test, y_pred)):
                print >> _fout,  "%s\t%d\t%f\t%d" % ('SHEFF/QuEst',i+1,_y[0],get_index(i,y_test))
        log.info('Done.')
    if y_test is None:
        X_train,y_train,X_test,y_test = prepare_data(x_train,y_train,x_test,y_test,separator=separator,scale=scale,flag=None)
        log.info('training ...')
        #是否需要优化模型参数
        if parameters == {}:
            pass
        else:
            clf = optimize_model(clf,X_train,y_train,parameters,scores,cv,True,-1)
        #训练模型
        log.info('Done.')
        log.info('Predict ...')
        y_pred = clf.predict(X_test)
        log.info('Done.')
        if flag == 'wmt16':
            y_pred = y_pred*100
        log.info('make result file ...')
        with open(predict_file, 'w') as _fout:
            for i, _y in enumerate(y_pred):
                print >> _fout,  "%s\t%d\t%f\t%d" % (team_name,i+1,_y,get_index(i,y_pred))
        log.info('Done.')
    
    log.info('耗时：%.1f min' % ((time.time()-start_time)/60))
    
    
if __name__ == '__main__':
    data_dir = '/new_home/czm/workspace/QE_project/data'
    out_dir = '/home/liutuan/czm/workspace/QE_project/learning'
    train(
        #特征和标签HTER
        #------------------------------------------------------------#
        x_train = data_dir+'/wmt16/task1/task1_en-de_training.baseline17.features',
        y_train = data_dir+'/wmt16/task1/train.hter',
        x_test = data_dir+'/wmt16/task1/task1_en-de_test.baseline17.features',
        y_test = data_dir+'/wmt16/task1/test.hter',
        flag = 'wmt16',
        cv = 3,
        scale = True,
        #------------------------------------------------------------#
        #parameters = {},
        parameters = {'C':[1,2,10],'epsilon':[0.1,0.2,2],'gamma':[0.0001,0.01,2]},
        #特征文件的分隔符
        separator = '\t',
        #预测输出的文件
        predict_file = out_dir+'/predicted.csv',
        #参考输出的文件
        ref_file = out_dir+'/ref.csv',
        #团队名称
        team_name = 'JXNU/word2vec+rnnlm',
        scores = ['Pearson']
        
    )
    
    
    
    
    
    
    
    
    
    
    
    
