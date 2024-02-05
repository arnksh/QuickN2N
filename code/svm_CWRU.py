# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 16:43:05 2021

@author: ARUN
"""

from __future__ import print_function

from sklearn import svm

import numpy as np

import scipy.io as sio

from sklearn import metrics
import xlsxwriter

#from sklearn.model_selection import train_test_split
allFolder = ['CWRU_40', 'CWRU_400']
tarData = ['FE_tar_7_1', 'FE_tar_7_2', 'FE_tar_7_3', 'FE_tar_14_1', 'FE_tar_14_2',
           'FE_tar_14_3', 'FE_tar_21_1', 'FE_tar_21_2', 'FE_tar_21_3', 'ims_tar'];
           
workbook = xlsxwriter.Workbook(f'./logs_CWRU/SVM_log.xlsx')
worksheet = workbook.add_worksheet()
# worksheet.write(1,4, f'{dataFolder}')
tr = 4
for dataFolder in allFolder:
    worksheet.write(1, tr, f'{dataFolder}')

    for i in range(len(tarData)):
        f0=sio.loadmat(f'../{dataFolder}/{tarData[i]}.mat')
        A = f0['Y'][0,0]
        x_train = A['training_inputs']
        y_train = A['training_results'][:,0]
        x_test = A['test_inputs']
        y_test = A['test_results'][:,0]
        
        if np.min(y_train)==1:
            y_train = y_train-1
            y_test = y_test-1
            
        clf = svm.SVC(kernel='rbf')
        
        clf.fit(x_train, y_train)

    #Predict the response for test dataset
        y_pred = clf.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred);
        worksheet.write(i+2,3, f'{tarData[i]}')
        worksheet.write(i+2,tr, accuracy*100)
    tr  = tr +1
    
workbook.close()


