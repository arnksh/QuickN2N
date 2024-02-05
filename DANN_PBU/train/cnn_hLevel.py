# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 22:16:53 2020

@author: AR-LAB
"""

import os
#import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np


def test(net, dataloader):
#    cuda = False
#    cudnn.benchmark = False
#    batch_size = 5
    image_size = 10
    alpha = 0

    net = net.eval()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    h_features = torch.tensor(np.array([]), dtype = torch.float32)

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 1, image_size, image_size)

        input_img.resize_as_(t_img).copy_(t_img)

        _, h_feature = net(input_data = input_img, alpha=alpha)
        h_features = torch.cat((h_features, h_feature),0)

        i += 1

    return h_features
