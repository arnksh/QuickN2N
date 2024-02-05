# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 17:51:09 2021

@author: ARUN
"""


from __future__ import print_function

from sklearn import svm

import numpy as np

import scipy.io as sio

from sklearn import metrics
import xlsxwriter

#from sklearn.model_selection import train_test_split
allFolder = ['PBU_400', 'PBU_40']
tarData = ['tarData_1','tarData_2','tarData_3','tarData_4'];
tarnames  = ['tar3', 'tar4']

workbook = xlsxwriter.Workbook(f'./logs_PBU/SVM_log.xlsx')
worksheet = workbook.add_worksheet()
tr = 4
for dataFolder in allFolder:
    worksheet.write(1, tr, f'{dataFolder}')
    
    j = 0
    for tar in tarnames:
        worksheet.write(j+2,2, f'{tar}')
        for i in range(len(tarData)):
            f0=sio.loadmat(f'../{dataFolder}/{tar}/{tarData[i]}.mat')
            A = f0['Y'][0,0]
            x_train = A['training_inputs']
            y_train = A['training_results'][:,0]
            x_test = A['test_inputs']
            y_test = A['test_results'][:,0]
            
            if np.min(y_train)==1:
                y_train = y_train-1
                y_test = y_test-1
                
            clf = svm.SVC(kernel='linear')
            
            clf.fit(x_train, y_train)
        
        #Predict the response for test dataset
            y_pred = clf.predict(x_test)
            accuracy = metrics.accuracy_score(y_test, y_pred);
            worksheet.write(j+2,3, f'{tarData[i]}')
            worksheet.write(j+2,tr, accuracy*100)
            j = j+1
    tr = tr+1
        
workbook.close()


