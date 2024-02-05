# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 18:59:36 2021

@author: ARUN
"""

import random
import os
from torch.utils import data
import torch.optim as optim
import torch.utils.data
#from torchvision import transforms
from models.model import CNNModel
#from models.tca import TCA
import numpy as np; import scipy.io as sio
from train.test import test
import xlsxwriter as xlw
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
#from libtlda.tca import TransferComponentClassifier
#from libtlda.iw import ImportanceWeightedClassifier
#from sklearn.metrics import accuracy_score
import time

dataFolder = 'PBU_40';
srcData  = 'sourceData';
tarData = ['tarData_1','tarData_2','tarData_3','tarData_4'];

model_root = os.path.join('.', 'models')
cuda = False
lr = 1e-2
image_size = 20
n_epoch = 40
manual_seed = random.randint(1, 10)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

workbook = xlw.Workbook(f'./logs/DANN_{dataFolder}.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(1,4, 'T1')
worksheet.write(1,5, 'T2')
#worksheet.write(1,5, 'Time (sec)')
#worksheet.write(1,7, 'Time (sec)')
#%%
InputDataSRC=sio.loadmat(f'../{dataFolder}/{srcData}.mat')
src_input = np.float32(np.array(InputDataSRC['Data'])) 
src_input1 = np.reshape(src_input, [src_input.shape[0],1, 20,20])
src_labels = np.array(InputDataSRC['labels'])[:,0]
dataset_src = data.TensorDataset(torch.tensor(src_input1), torch.tensor(src_labels, dtype=torch.long))
loader_src = data.DataLoader(dataset=dataset_src, 
                                 batch_size=4, num_workers=0,
                                 shuffle=True)

for T in [3,4]:
    for mload in range(len(tarData)):
        f0 = sio.loadmat(f'../{dataFolder}/tar{T}/{tarData[mload]}.mat')
        InputDataTAR = f0['Y'][0,0]
        tar_input = np.float32(np.array(InputDataTAR['training_inputs']))
        tar_input1 = np.reshape(tar_input, [tar_input.shape[0],1, 20,20])
        tar_labels = np.array(InputDataTAR['training_results'])[:,0]
    
        dataset_tar =  data.TensorDataset(torch.tensor(tar_input1), torch.tensor(tar_labels,dtype=torch.long))
        loader_tar = data.DataLoader(dataset = dataset_tar,
                                     batch_size=4, num_workers=0,
                                     shuffle=True)
    
        # create test loader
        test_tar_input = np.float32(np.array(InputDataTAR['test_inputs']))
        test_tar_input1 = np.reshape(test_tar_input, [test_tar_input.shape[0],1, 20,20])
        test_tar_labels = np.array(InputDataTAR['test_results'])[:,0]
        
        test_dataset_tar =  data.TensorDataset(torch.tensor(test_tar_input1), torch.tensor(test_tar_labels,dtype=torch.long))
        test_loader_tar = data.DataLoader(dataset = test_dataset_tar,
                                     batch_size=4, num_workers=0,
                                     shuffle=True)
    #%%
        my_net = CNNModel(imageSize = image_size)
        optimizer = optim.Adam(my_net.parameters(), lr=lr)
        loss_class = torch.nn.NLLLoss()
        loss_domain = torch.nn.NLLLoss()
         
        for p in my_net.parameters():
            p.requires_grad = True
    
        # training
        startTime = time.time() # Start  Training
        for epoch in range(n_epoch):
            len_dataloader = min(len(loader_src), len(loader_tar))
            data_source_iter = iter(loader_src)
            data_target_iter = iter(loader_tar) #input,label=next(iter(TrainLoader))
        
            i = 0
            while i < len_dataloader:
        
                p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
                # training model using source data
                s_img, s_label = next(data_source_iter)
        
                my_net.zero_grad()
                batch_size = len(s_label)
        
                input_img = torch.FloatTensor(batch_size, 1, image_size, image_size)
                class_label = torch.LongTensor(batch_size)
                domain_label = torch.zeros(batch_size)
                domain_label = domain_label.long()
        
                input_img.resize_as_(s_img).copy_(s_img)
                class_label.resize_as_(s_label).copy_(s_label)
        
                class_output, _,  domain_output = my_net(input_data=input_img, alpha=alpha)
                err_s_label = loss_class(class_output, class_label)
                err_s_domain = loss_domain(domain_output, domain_label)
        
                # training model using target data
                t_img, _ = next(data_target_iter)
        
                batch_size = len(t_img)
        
                input_img = torch.FloatTensor(batch_size, 1, image_size, image_size)
                domain_label = torch.ones(batch_size)
                domain_label = domain_label.long()
        
                input_img.resize_as_(t_img).copy_(t_img)
        
                _, _, domain_output = my_net(input_data=input_img, alpha=alpha)
                err_t_domain = loss_domain(domain_output, domain_label)
                err = err_t_domain + err_s_domain + err_s_label
                err.backward()
                optimizer.step()
        
                i += 1
        
            torch.save(my_net, '{0}/{1}_model_epoch_{2}.pth'.format(model_root, dataFolder, epoch))
            if epoch ==99:
                src_acc, _, _ = test(srcData, loader_src, epoch)
                tar_acc, _, _ = test(tarData[mload], loader_tar, epoch)
                print ('epoch: %d, accuracy on the %s data: %f and %s data: %f' % (epoch, srcData, 
                                                                                   src_acc, tarData[mload], tar_acc))
        endTime = time.time()   # End Training 
        acc, h_features, predLabels = test(tarData[mload], test_loader_tar, 1)
        h_features = h_features.detach().numpy()
        predLabels = predLabels.detach().numpy()
        sio.savemat(f'./DANN_hdata/{dataFolder}_{tarData[mload]}.mat', {'hTestData': h_features, 'predLabels':predLabels})
        worksheet.write(mload+2,3, tarData[mload])
        worksheet.write(mload+2,T+1, acc*100)
#        worksheet.write(mload+2,T+2, (endTime-startTime))
        
        
    
    
    
#    clf = TransferComponentClassifier()
#    clf.fit(src_input, src_labels, tar_input)
#    acc = accuracy_score(test_tar_labels, clf.predict(test_tar_input))
#    worksheet.write(mload+5,2, f'{acc*100}')
#    
#    clf = ImportanceWeightedClassifier()
#    clf.fit(src_input, src_labels, tar_input)
#    acc = accuracy_score(test_tar_labels, clf.predict(test_tar_input))
#    worksheet.write(mload+5,3, f'{acc*100}')
#    
#    tca = TCA(kernel_type='linear', dim=20, lamb=1, gamma=1) # kernel = 'rbf' may be used
#    acc, ypred = tca.fit_predict(src_input, src_labels, tar_input, tar_labels)
#    acc
        
#    svm_model = SVC(kernel='rbf', C =10, gamma=0.02).fit(tar_input, tar_labels)  # for rbf
#    acc_svm = svm_model.score(test_tar_input, test_tar_labels)
#    worksheet.write(mload+5,2, f'{acc_svm*100}')
#    
#    rf_model=RandomForestClassifier(n_estimators = 100).fit(tar_input, tar_labels)
#    acc_rf=rf_model.score(test_tar_input, test_tar_labels)
#    worksheet.write(mload+5,3, f'{acc_rf*100}')
  
workbook.close()
